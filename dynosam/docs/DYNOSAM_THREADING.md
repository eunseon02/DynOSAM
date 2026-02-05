# Pipeline Threading Architecture (파이프라인 스레딩 아키텍처)

## 개요

DynoSAM 파이프라인은 frontend와 backend 처리를 병렬화하기 위해 멀티스레드 아키텍처를 사용합니다. 최적의 성능을 달성하기 위해 여러 레벨에서 스레딩이 관리됩니다.

## 스레딩 레벨

### 1. Pipeline 레벨 스레딩 (PipelineManager)

`params_.parallelRun()`이 `true`일 때, `PipelineManager`는 `Spinner` 클래스를 사용하여 frontend와 backend 파이프라인을 위한 별도 스레드를 생성합니다.

**위치**: `dynosam/src/pipeline/PipelineManager.cc`의 `PipelineManager::launchSpinners()`

**구성 요소**:
- **FrontendPipeline**: `frontend_pipeline_spinner_`를 통해 자체 스레드에서 실행
- **BackendPipeline**: `backend_pipeline_spinner_`를 통해 자체 스레드에서 실행
- **DataInterfacePipeline**: 항상 별도 스레드에서 실행
- **Visualization pipelines**: 별도 스레드에서 실행 (`frontend_viz_pipeline_spinner_`, `backend_viz_pipeline_spinner_`)

**구현**:
```cpp
if (params_.parallelRun()) {
  frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
      std::bind(&dyno::FrontendPipeline::spin, frontend_pipeline_.get()),
      "frontend-pipeline-spinner");

  if (backend_pipeline_)
    backend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::BackendPipeline::spin, backend_pipeline_.get()),
        "backend-pipeline-spinner");
}
```

각 `Spinner`는 `PipelineBase::spin()`을 `std::thread`로 감싸서, 종료될 때까지 `spinOnce()`를 호출하는 연속 루프를 생성합니다.

### 2. Queue 레벨 스레딩 (SIMOPipelineModule)

각 파이프라인 모듈(`FrontendPipeline`, `BackendPipeline`)은 `SIMOPipelineModule`을 상속받으며, 입력 큐 접근을 관리합니다.

**위치**: `dynosam/include/dynosam/pipeline/PipelineBase-inl.hpp`의 `SIMOPipelineModule::getInputPacket()`

**모드**:
- **Blocking 모드** (`parallel_run_ = true`): `popBlocking()` 사용 - 데이터를 기다리며 블로킹 (전용 스레드에 적합)
- **Non-blocking 모드** (`parallel_run_ = false`): `pop()` 사용 - 논블로킹 (동기 실행에 적합)

**구현**:
```cpp
if (parallel_run_) {
  queue_state = input_queue->popBlocking(input);  // Blocking (전용 스레드용)
} else {
  queue_state = input_queue->pop(input);  // Non-blocking (동기 실행용)
}
```

### 3. Module 레벨 병렬 처리 (BackendModule)

Backend 모듈 내부(예: `ParallelHybridBackendModule`)에서는 **Intel TBB (Threading Building Blocks)**를 사용하여 추가 병렬 처리를 수행합니다.

**위치**: `dynosam/src/backend/ParallelHybridBackendModule.cc`의 `ParallelHybridBackendModule::parallelObjectSolve()`

**병렬 작업**:
- **Object 레벨 병렬 처리**: `tbb::parallel_for_each`가 여러 객체를 동시에 처리
- **Edge 처리**: `tbb::parallel_for`가 edge를 병렬로 처리 (`KeyFrame.cc`에서)
- **최적화**: `tbb::parallel_reduce`를 사용한 병렬 비용 계산 (`Optimizer.cc`에서)

**예제**:
```cpp
tbb::parallel_for_each(
    object_tracks.begin(), object_tracks.end(),
    [&](const std::pair<ObjectId, VisionImuPacket::ObjectTracks>& update) {
      this->implSolvePerObject(frame_id, update.first, update.second, X_W_k);
    });
```

### 4. Frontend 내부 스레딩

Frontend 모듈(`RGBDInstanceFrontendModule`)은 특정 작업을 위해 내부 스레딩을 가질 수 있습니다:

**Edge 최적화 스레드**: 프레임 처리를 블로킹하지 않기 위해 별도 스레드가 sliding window 최적화를 처리합니다.

**위치**: `dynosam/src/frontend/RGBDInstanceFrontendModule.cc`의 `RGBDInstanceFrontendModule::optimizationThreadFunction()`

## 실행 흐름

### `parallelRun()`이 활성화된 경우

```
Main Thread (메인 스레드)
├─ PipelineManager::spin()
│   └─ 파이프라인이 작동 중인지 확인 (논블로킹)
│
Frontend Thread (Spinner)
└─ FrontendPipeline::spin()
    └─ while (!isShutdown())
        └─ FrontendPipeline::spinOnce()
            ├─ getInputPacket() [큐에서 popBlocking]
            ├─ process() → FrontendModule::spinOnce()
            │   └─ 내부 스레드 트리거 가능 (예: edge 최적화)
            └─ pushOutputPacket() → BackendPipeline 입력 큐

Backend Thread (Spinner)
└─ BackendPipeline::spin()
    └─ while (!isShutdown())
        └─ BackendPipeline::spinOnce()
            ├─ getInputPacket() [큐에서 popBlocking]
            ├─ process() → BackendModule::spinOnce()
            │   └─ TBB parallel_for_each (object 레벨 병렬 처리)
            │       └─ 각 객체가 병렬로 처리됨
            └─ pushOutputPacket()
```

### `parallelRun()`이 비활성화된 경우

```
Main Thread (메인 스레드)
└─ PipelineManager::spin()
    └─ 순차적으로 호출:
        ├─ FrontendPipeline::spinOnce()
        └─ BackendPipeline::spinOnce()
```

## 스레드 안전성

### 스레드 간 통신

- **스레드 안전 큐** (`ThreadsafeQueue`)가 파이프라인 간 통신에 사용됩니다
- 각 파이프라인은 자체 입력/출력 큐로 독립적으로 작동합니다
- 데이터는 `shared_ptr<const T>`를 통해 전달되어 안전한 동시 접근을 보장합니다

### BackendModule 내부 병렬 처리

- **TBB**는 사용 가능한 CPU 코어 수에 따라 스레드 풀 크기를 자동으로 관리합니다
- TBB는 명시적인 mutex 관리 없이 스레드 안전한 병렬 실행을 제공합니다
- 각 병렬 작업은 독립적인 데이터 구조에서 작동합니다

### Frontend 내부 스레딩

- **Mutex 보호**: `local_map_mutex_`가 공유 데이터 구조를 보호합니다
- **Atomic 플래그**: `optimization_in_progress_`가 동시 최적화를 방지합니다
- **조건 변수**: 스레드 동기화 및 깨우기에 사용됩니다

## 성능 고려사항

### CPU 사용량

- `PipelineBase::spin()`의 **5ns sleep**은 작업이 없을 때 CPU spinning을 방지합니다
- 블로킹 큐 작업(`popBlocking`)은 유휴 상태일 때 스레드가 sleep하도록 하여 CPU 사용량을 줄입니다
- TBB는 사용 가능한 코어에 걸쳐 워크로드를 자동으로 균형 조정합니다

### Lock 경합

- Lock 경합은 다음을 통해 최소화됩니다:
  - 각 파이프라인에 별도 큐 사용
  - 상태 추적을 위한 atomic 플래그
  - Lock 범위 최소화 (필요할 때만 lock)
  - 거친 lock 대신 세밀한 병렬 처리를 위한 TBB 사용

### 처리량

- **병렬 실행**은 frontend와 backend가 데이터를 동시에 처리할 수 있게 합니다
- `parallelRun()`이 활성화되면 메인 스레드에서 **논블로킹 작업** 수행
- **내부 병렬 처리** (TBB)는 계산 집약적인 작업에 대해 CPU 활용을 최대화합니다

## 설정

스레딩 동작은 `PipelineParams`의 `parallel_run` 파라미터로 제어됩니다:

```cpp
struct PipelineParams {
  bool parallel_run{true};  // Pipeline 레벨 스레딩 활성화/비활성화
};
```

이 파라미터는 다음을 통해 설정됩니다:
- YAML 설정 파일
- 명령줄 플래그
- 프로그래밍 방식 설정

## 관련 파일

- **Pipeline 관리**: `dynosam/src/pipeline/PipelineManager.cc`
- **Pipeline 기본**: `dynosam/src/pipeline/PipelineBase.cc`
- **Spinner 구현**: `dynosam_common/src/utils/Spinner.cc`
- **Backend 병렬 처리**: `dynosam/src/backend/ParallelHybridBackendModule.cc`
- **Frontend 스레딩**: `dynosam/src/frontend/RGBDInstanceFrontendModule.cc`
- **큐 구현**: `dynosam/include/dynosam/pipeline/ThreadSafeQueue.hpp`

## 참고사항

- 스레딩 아키텍처는 **확장 가능**하고 **설정 가능**하도록 설계되었습니다
- 스레드 안전성은 스레드 안전 데이터 구조와 동기화 기본 요소를 신중하게 사용하여 보장됩니다
- 성능은 `parallel_run`과 TBB 스레드 풀 크기를 조정하여 튜닝할 수 있습니다
- 아키텍처는 단일 스레드(디버깅용) 및 멀티스레드(프로덕션용) 실행 모드를 모두 지원합니다

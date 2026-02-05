#include "dynosam/visualizer/TrajectoryLoggerDisplay.hpp"
#include <glog/logging.h>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dyno {

TrajectoryLoggerDisplay::TrajectoryLoggerDisplay(const std::string& output_file, bool show_window)
    : output_file_(output_file), show_window_(show_window), 
      viewer_running_(false),
      mViewpointX(0.0f), mViewpointY(-0.7f), mViewpointZ(-2.0f), mViewpointF(500.0f),
      mbFollowCamera(true), mbShowTrajectory(true), mbShowKeyFrames(false), mbShowGraph(false) {
  trajectory_file_.open(output_file_);
  if (!trajectory_file_.is_open()) {
    LOG(WARNING) << "Failed to open trajectory output file: " << output_file_;
  } else {
    LOG(INFO) << "Trajectory logging to: " << output_file_;
  }
  
  if (show_window_) {
    viewer_running_ = true;
    viewer_thread_ = std::thread(&TrajectoryLoggerDisplay::runPangolinViewer, this);
    LOG(INFO) << "Pangolin 3D viewer thread started";
  }
}

TrajectoryLoggerDisplay::~TrajectoryLoggerDisplay() {
  if (trajectory_file_.is_open()) {
    trajectory_file_.close();
    LOG(INFO) << "Trajectory saved to: " << output_file_ << " (total poses: " << trajectory_.size() << ")";
  }
  
  if (show_window_ && viewer_running_) {
    viewer_running_ = false;
    if (viewer_thread_.joinable()) {
      viewer_thread_.join();
    }
    LOG(INFO) << "Pangolin viewer thread stopped";
  }
}

void TrajectoryLoggerDisplay::runPangolinViewer() {
  // Disable X11 shared memory to avoid MESA errors in Docker/remote environments
  // This may cause slight performance degradation but ensures rendering works
  setenv("LIBGL_ALWAYS_INDIRECT", "1", 0);
  setenv("MESA_GL_VERSION_OVERRIDE", "3.3", 0);
  // Disable X11 shared memory extension to avoid MESA errors
  setenv("LIBGL_ALWAYS_SOFTWARE", "0", 0);
  setenv("GALLIUM_DRIVER", "llvmpipe", 0);
  
  // Initialize Pangolin
  pangolin::CreateWindowAndBind("DynoSAM: Trajectory Viewer", 1024, 768);
  
  // 3D Mouse handler requires depth testing to be enabled
  glEnable(GL_DEPTH_TEST);
  
  // Issue specific OpenGL we might need
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
  // Set initial clear color
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  
  // Create menu panel
  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", mbFollowCamera, true);
  pangolin::Var<bool> menuShowTrajectory("menu.Show Trajectory", mbShowTrajectory, true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", mbShowKeyFrames, true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph", mbShowGraph, true);
  pangolin::Var<bool> menuQuit("menu.Quit", false, false);
  
  // Define camera render object
  // Make sure camera can see the origin clearly
  // Use a simple, close-up view to ensure we see something
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
  );
  
  // Log OpenGL info for debugging
  LOG(INFO) << "OpenGL Version: " << glGetString(GL_VERSION);
  LOG(INFO) << "OpenGL Vendor: " << glGetString(GL_VENDOR);
  LOG(INFO) << "OpenGL Renderer: " << glGetString(GL_RENDERER);
  
  // Force initial camera matrix to ensure we're looking at origin
  s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
  
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));
  
  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();
  
  bool bFollow = true;
  
  LOG(INFO) << "========================================";
  LOG(INFO) << "Pangolin viewer loop started";
  LOG(INFO) << "Initial camera position: (" << mViewpointX << ", " << mViewpointY << ", " << mViewpointZ << ")";
  LOG(INFO) << "Initial camera focal: " << mViewpointF;
  LOG(INFO) << "Window created: DynoSAM: Trajectory Viewer (1024x768)";
  LOG(INFO) << "========================================";
  
  int frame_count = 0;
  while (viewer_running_ && !pangolin::ShouldQuit()) {
    // Get latest trajectory for rendering
    std::vector<std::pair<double, gtsam::Pose3>> current_trajectory;
    {
      std::lock_guard<std::mutex> lock(trajectory_mutex_);
      current_trajectory = trajectory_;
    }
    
    // Log trajectory status periodically
    if (frame_count == 0) {
      LOG(INFO) << "========================================";
      LOG(INFO) << "DIAGNOSTIC: First frame rendering";
      LOG(INFO) << "  - Trajectory size: " << current_trajectory.size();
      LOG(INFO) << "  - Camera position: (" << mViewpointX << ", " << mViewpointY << ", " << mViewpointZ << ")";
      LOG(INFO) << "  - Camera focal: " << mViewpointF;
      LOG(INFO) << "  - Looking at: (0, 0, 0)";
      if (!current_trajectory.empty()) {
        const auto& last_pose = current_trajectory.back().second;
        const auto& t = last_pose.translation();
        LOG(INFO) << "  - Last trajectory point: (" << t.x() << ", " << t.y() << ", " << t.z() << ")";
        LOG(INFO) << "  - DIAGNOSIS: Trajectory data EXISTS - rendering should work";
      } else {
        LOG(INFO) << "  - No trajectory data yet (will draw test objects)";
        LOG(INFO) << "  - DIAGNOSIS: Trajectory data MISSING - checking if spin() is being called";
      }
      LOG(INFO) << "========================================";
    }
    if (frame_count % 100 == 0 && frame_count > 0) {
      LOG(INFO) << "Viewer frame " << frame_count << " - trajectory size=" << current_trajectory.size();
      if (!current_trajectory.empty()) {
        const auto& last_pose = current_trajectory.back().second;
        const auto& t = last_pose.translation();
        LOG(INFO) << "  Last pose translation: (" << t.x() << ", " << t.y() << ", " << t.z() << ")";
      } else {
        LOG(WARNING) << "  WARNING: Still no trajectory data after " << frame_count << " frames!";
        LOG(WARNING) << "  This suggests spin() is not being called or trajectory is not being updated";
      }
    }
    frame_count++;
    
    // Get current camera pose if available
    if (!current_trajectory.empty()) {
      const gtsam::Pose3& current_pose = current_trajectory.back().second;
      const gtsam::Matrix4& T_matrix = current_pose.matrix();
      
      // Convert gtsam::Pose3 to pangolin::OpenGlMatrix
      // Pangolin uses column-major, gtsam uses row-major
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          Twc.m[j * 4 + i] = T_matrix(i, j);
        }
      }
    }
    
    // Handle follow camera
    if (!current_trajectory.empty()) {
      if (menuFollowCamera && bFollow) {
        s_cam.Follow(Twc);
      } else if (menuFollowCamera && !bFollow) {
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
        s_cam.Follow(Twc);
        bFollow = true;
      } else if (!menuFollowCamera && bFollow) {
        bFollow = false;
      }
    } else {
      // If no trajectory yet, use default view
      if (!bFollow) {
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
      }
    }
    
    // Update menu variables
    mbFollowCamera = menuFollowCamera;
    mbShowTrajectory = menuShowTrajectory;
    mbShowKeyFrames = menuShowKeyFrames;
    mbShowGraph = menuShowGraph;
    
    // Clear the screen FIRST (exactly like Viewer.cc line 130)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Set clear color BEFORE activate
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    
    // Force viewport to ensure we're rendering to the right area
    // Check viewport BEFORE and AFTER Activate to diagnose issue
    GLint viewport_before[4];
    glGetIntegerv(GL_VIEWPORT, viewport_before);
    
    // Activate display (exactly like Viewer.cc line 160)
    d_cam.Activate(s_cam);
    
    GLint viewport_after[4];
    glGetIntegerv(GL_VIEWPORT, viewport_after);
    
    if (frame_count < 3) {
      LOG(INFO) << "DIAGNOSTIC: OpenGL State";
      LOG(INFO) << "  - Viewport BEFORE Activate: [" << viewport_before[0] << ", " << viewport_before[1] 
                << ", " << viewport_before[2] << ", " << viewport_before[3] << "]";
      LOG(INFO) << "  - Viewport AFTER Activate: [" << viewport_after[0] << ", " << viewport_after[1] 
                << ", " << viewport_after[2] << ", " << viewport_after[3] << "]";
      GLfloat clear_color[4];
      glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color);
      LOG(INFO) << "  - Clear color: (" << clear_color[0] << ", " << clear_color[1] 
                << ", " << clear_color[2] << ", " << clear_color[3] << ")";
      
      // Check if viewport is reasonable (should be [175, 0, 849, 768] for menu width 175)
      if (viewport_after[2] < 100 || viewport_after[3] < 100 || viewport_after[0] < 0 || viewport_after[1] < 0) {
        LOG(ERROR) << "  - WARNING: Viewport is invalid! This will cause rendering to fail!";
        LOG(ERROR) << "  - Expected: [175, 0, 849, 768] (menu=175px, window=1024x768)";
      } else {
        LOG(INFO) << "  - Viewport is valid";
      }
    }
    
    // IMPORTANT: Clear again after Activate to ensure background is visible
    // This is critical for the first frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Check for OpenGL errors
    GLenum err = glGetError();
    if (frame_count < 3) {
      if (err != GL_NO_ERROR) {
        LOG(WARNING) << "Frame " << frame_count << " - OpenGL error after Activate: " << err;
      } else {
        LOG(INFO) << "Frame " << frame_count << " - OpenGL OK after Activate";
      }
    }
    
    // Draw trajectory
    if (mbShowTrajectory && !current_trajectory.empty()) {
      drawTrajectory3D();
    }
    
    // Draw current camera
    if (!current_trajectory.empty()) {
      glPushMatrix();
      glMultMatrixd(Twc.m);
      
      // Draw camera as a pyramid
      const float w = 0.08f;
      const float h = w * 0.75f;
      const float z = w * 0.6f;
      
      glLineWidth(2.0f);
      glColor3f(0.0f, 0.0f, 1.0f);  // Blue
      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(w, h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, -h, z);
      glVertex3f(0, 0, 0);
      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);
      glVertex3f(w, -h, z);
      glVertex3f(-w, h, z);
      glVertex3f(-w, -h, z);
      glVertex3f(-w, h, z);
      glVertex3f(w, h, z);
      glVertex3f(-w, -h, z);
      glVertex3f(w, -h, z);
      glEnd();
      
      glPopMatrix();
    }
    
    // ALWAYS DRAW TEST OBJECTS - These should be visible even with empty trajectory
    // Draw MASSIVE objects that fill the entire view - IMPOSSIBLE TO MISS
    
    if (frame_count <= 2) {
      LOG(INFO) << "DIAGNOSTIC: Rendering test objects at frame " << frame_count;
      LOG(INFO) << "  - Will draw: 2D NDC overlay (white bg, red rect, green circle, blue cross, yellow bar)";
      LOG(INFO) << "  - Will draw: 3D objects (red plane, axes, grid)";
    }
    
    // Draw in NDC (Normalized Device Coordinates) space to ensure visibility
    // This bypasses any camera/view matrix issues and draws directly to screen
    
    // Save current matrices and state
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);  // Explicit orthographic projection for NDC
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    // Save and modify OpenGL state
    GLboolean depth_test_enabled;
    glGetBooleanv(GL_DEPTH_TEST, &depth_test_enabled);
    glDisable(GL_DEPTH_TEST);
    
    // Draw a MASSIVE bright red rectangle covering 90% of screen - IMPOSSIBLE TO MISS
    // Use glColor4f with alpha to ensure it's visible
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);  // Bright red, fully opaque
    glBegin(GL_QUADS);
    glVertex2f(-0.9f, -0.9f);
    glVertex2f(0.9f, -0.9f);
    glVertex2f(0.9f, 0.9f);
    glVertex2f(-0.9f, 0.9f);
    glEnd();
    
    // Draw a full-screen white rectangle first to ensure background is visible
    // This should ALWAYS be visible regardless of camera/viewpoint
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);  // White - should fill entire screen
    glBegin(GL_QUADS);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f(1.0f, -1.0f);
    glVertex2f(1.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();
    
    // Draw red rectangle on top - should be clearly visible
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);  // Bright red
    glBegin(GL_QUADS);
    glVertex2f(-0.9f, -0.9f);
    glVertex2f(0.9f, -0.9f);
    glVertex2f(0.9f, 0.9f);
    glVertex2f(-0.9f, 0.9f);
    glEnd();
    
    if (frame_count <= 2) {
      // Verify the drawing commands were issued
      GLenum err = glGetError();
      if (err != GL_NO_ERROR) {
        LOG(WARNING) << "OpenGL error after drawing white/red rectangles: " << err;
      } else {
        LOG(INFO) << "  - White background and red rectangle drawn successfully";
      }
    }
    
    // Draw a bright green filled circle in center
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);  // Bright green, fully opaque
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(0, 0);  // Center
    const int segments = 64;
    const float radius = 0.4f;  // Larger radius
    for (int i = 0; i <= segments; i++) {
      float angle = 2.0f * M_PI * i / segments;
      glVertex2f(radius * cos(angle), radius * sin(angle));
    }
    glEnd();
    
    // Draw a bright blue thick cross
    glLineWidth(20.0f);
    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);  // Bright blue, fully opaque
    glBegin(GL_LINES);
    glVertex2f(-0.6f, 0);
    glVertex2f(0.6f, 0);
    glVertex2f(0, -0.6f);
    glVertex2f(0, 0.6f);
    glEnd();
    
    // Draw yellow text-like rectangle at top
    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow, fully opaque
    glBegin(GL_QUADS);
    glVertex2f(-0.8f, 0.7f);
    glVertex2f(0.8f, 0.7f);
    glVertex2f(0.8f, 0.9f);
    glVertex2f(-0.8f, 0.9f);
    glEnd();
    
    // Restore state
    if (depth_test_enabled) {
      glEnable(GL_DEPTH_TEST);
    }
    
    // Restore matrices
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    // Now draw 3D objects in world space
    // 1. Draw a HUGE filled red rectangle on ground plane
    glColor3f(1.0f, 0.5f, 0.5f);  // Light red
    glBegin(GL_QUADS);
    glVertex3f(-5.0, -5.0, 0);
    glVertex3f(5.0, -5.0, 0);
    glVertex3f(5.0, 5.0, 0);
    glVertex3f(-5.0, 5.0, 0);
    glEnd();
    
    if (frame_count <= 2) {
      LOG(INFO) << "DIAGNOSTIC: All test objects drawn";
      LOG(INFO) << "  - 2D NDC overlay: white bg, red rect(90%), green circle, blue cross, yellow bar";
      LOG(INFO) << "  - 3D world space: red plane, axes, grid";
      GLenum err = glGetError();
      if (err != GL_NO_ERROR) {
        LOG(WARNING) << "  - OpenGL error after drawing all objects: " << err;
      } else {
        LOG(INFO) << "  - No OpenGL errors, all drawing commands executed";
      }
    }
    
    // Draw coordinate axes at origin (always visible) - VERY LARGE
    glLineWidth(8.0f);
    glBegin(GL_LINES);
    // X axis - Red (very long for visibility)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(2.0, 0, 0);
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 2.0, 0);
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 2.0);
    glEnd();
    
    // Draw a large filled triangle - VERY VISIBLE
    glColor3f(1.0f, 1.0f, 0.0f);  // Yellow - very bright
    glBegin(GL_TRIANGLES);
    glVertex3f(0.5, 0, 0);
    glVertex3f(0, 0.5, 0);
    glVertex3f(0, 0, 0.5);
    glEnd();
    
    // Draw a large cube wireframe at origin
    glLineWidth(4.0f);
    glColor3f(1.0f, 0.0f, 1.0f);  // Magenta
    const float cube_size = 0.8f;
    glBegin(GL_LINE_LOOP);
    glVertex3f(-cube_size, -cube_size, cube_size);
    glVertex3f(cube_size, -cube_size, cube_size);
    glVertex3f(cube_size, cube_size, cube_size);
    glVertex3f(-cube_size, cube_size, cube_size);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex3f(-cube_size, -cube_size, -cube_size);
    glVertex3f(cube_size, -cube_size, -cube_size);
    glVertex3f(cube_size, cube_size, -cube_size);
    glVertex3f(-cube_size, cube_size, -cube_size);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-cube_size, -cube_size, cube_size);
    glVertex3f(-cube_size, -cube_size, -cube_size);
    glVertex3f(cube_size, -cube_size, cube_size);
    glVertex3f(cube_size, -cube_size, -cube_size);
    glVertex3f(cube_size, cube_size, cube_size);
    glVertex3f(cube_size, cube_size, -cube_size);
    glVertex3f(-cube_size, cube_size, cube_size);
    glVertex3f(-cube_size, cube_size, -cube_size);
    glEnd();
    
    // Draw a grid on the ground plane (always visible)
    glLineWidth(2.0f);
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    for (int i = -5; i <= 5; i++) {
      glVertex3f(i, -5, 0);
      glVertex3f(i, 5, 0);
      glVertex3f(-5, i, 0);
      glVertex3f(5, i, 0);
    }
    glEnd();
    
    // Check for OpenGL errors before finishing frame
    err = glGetError();
    if (frame_count < 3) {
      if (err != GL_NO_ERROR) {
        LOG(WARNING) << "Frame " << frame_count << " - OpenGL error before FinishFrame: " << err;
      } else {
        LOG(INFO) << "Frame " << frame_count << " - OpenGL OK before FinishFrame, drawing " 
                  << (current_trajectory.empty() ? "test objects only" : "trajectory + test objects");
        // Log what we actually drew
        LOG(INFO) << "Frame " << frame_count << " - Objects drawn: 2D overlay (NDC space) + 3D objects";
      }
    }
    
    // Force flush before finishing frame
    glFlush();
    
    pangolin::FinishFrame();
    
    // Verify frame was rendered
    if (frame_count == 1) {
      LOG(INFO) << "========================================";
      LOG(INFO) << "First frame rendered and swapped to screen";
      LOG(INFO) << "Rendered objects:";
      LOG(INFO) << "  - 2D NDC overlay: red rect(80% screen), green circle, blue cross";
      LOG(INFO) << "  - 3D world space: red plane, axes, grid";
      LOG(INFO) << "If you see black screen, check:";
      LOG(INFO) << "  1. DISPLAY environment variable: " << (getenv("DISPLAY") ? getenv("DISPLAY") : "NOT SET");
      LOG(INFO) << "  2. X11 server is running and accessible";
      LOG(INFO) << "  3. Window is visible (check other desktops/windows)";
      LOG(INFO) << "  4. Try moving mouse over window area";
      LOG(INFO) << "========================================";
    }
    
    if (menuQuit) {
      viewer_running_ = false;
    }
    
    // Small sleep to avoid busy waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  pangolin::DestroyWindow("DynoSAM: Trajectory Viewer");
  LOG(INFO) << "Pangolin viewer loop ended";
}

void TrajectoryLoggerDisplay::drawTrajectory3D() {
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  
  if (trajectory_.size() < 2) {
    return;
  }
  
  // Draw trajectory as lines
  glLineWidth(2.0f);
  glColor3f(0.0f, 1.0f, 0.0f);  // Green
  glBegin(GL_LINE_STRIP);
  
  for (const auto& [timestamp, pose] : trajectory_) {
    const gtsam::Point3& t = pose.translation();
    glVertex3f(t.x(), t.y(), t.z());
  }
  
  glEnd();
  
  // Draw keyframes as small spheres (if enabled)
  if (mbShowKeyFrames) {
    glPointSize(5.0f);
    glColor3f(1.0f, 0.0f, 0.0f);  // Red
    glBegin(GL_POINTS);
    
    for (const auto& [timestamp, pose] : trajectory_) {
      const gtsam::Point3& t = pose.translation();
      glVertex3f(t.x(), t.y(), t.z());
    }
    
    glEnd();
  }
  
  // Draw graph edges (if enabled)
  if (mbShowGraph && trajectory_.size() > 1) {
    glLineWidth(1.0f);
    glColor3f(0.5f, 0.5f, 0.5f);  // Gray
    glBegin(GL_LINES);
    
    for (size_t i = 1; i < trajectory_.size(); ++i) {
      const gtsam::Point3& t1 = trajectory_[i-1].second.translation();
      const gtsam::Point3& t2 = trajectory_[i].second.translation();
      glVertex3f(t1.x(), t1.y(), t1.z());
      glVertex3f(t2.x(), t2.y(), t2.z());
    }
    
    glEnd();
  }
}

void TrajectoryLoggerDisplay::spin(const BackendOutputPacket::ConstPtr& output) {
  if (!output) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: received null output";
    return;
  }
  
  if (!trajectory_file_.is_open()) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: trajectory file not open";
    return;
  }

  // Get current optimized pose
  const gtsam::Pose3& pose = output->T_world_camera;
  const double timestamp = output->timestamp;
  
  // Check if pose is valid (not identity/zero)
  const gtsam::Point3& t = pose.translation();
  const double translation_norm = t.norm();
  
  if (translation_norm < 1e-6) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: received zero/identity pose at frame=" 
                 << output->frame_id << ", timestamp=" << timestamp;
  }
  
  static int spin_count = 0;
  spin_count++;
  if (spin_count <= 5 || spin_count % 50 == 0) {
    LOG(INFO) << "DIAGNOSTIC: spin() called - frame=" << output->frame_id 
              << ", timestamp=" << timestamp
              << ", pose=(" << t.x() << ", " << t.y() << ", " << t.z() << ")"
              << ", spin_count=" << spin_count;
  }

  // Save to file immediately (TUM format: timestamp tx ty tz qx qy qz qw)
  const gtsam::Rot3& R = pose.rotation();
  const gtsam::Quaternion q = R.toQuaternion();

  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_file_ << std::fixed << std::setprecision(6) 
                     << timestamp << " "
                     << t.x() << " " << t.y() << " " << t.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    trajectory_file_.flush();  // Flush immediately for real-time monitoring
    
    // Also store in memory
    if (spin_count <= 5 || spin_count % 50 == 0) {
      LOG(INFO) << "  - Trajectory updated in memory, new size=" << trajectory_.size();
    }
    trajectory_.emplace_back(timestamp, pose);
    
    // Log every 10 frames for monitoring
    if (trajectory_.size() % 10 == 0) {
      LOG(INFO) << "Trajectory logged: frame=" << output->frame_id 
                << ", timestamp=" << timestamp 
                << ", total_poses=" << trajectory_.size();
    }
  }
}

std::vector<std::pair<double, gtsam::Pose3>> TrajectoryLoggerDisplay::getTrajectory() const {
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  return trajectory_;
}

}  // namespace dyno

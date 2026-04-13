#include "dynosam/visualizer/VoViewer.hpp"
#include "dynosam/backend/edge_map/Map.hpp"
#include "dynosam/frontend/vision/Object.hpp"

// Local toggle for raw object edges (most recent KeyFrame), kept file-scoped to avoid header changes.
static std::shared_ptr<pangolin::Var<bool>> menuShowRawObjectEdges;

pangolin::OpenGlMatrix Eigen2gl(Eigen::Matrix4f matrix)
{
    pangolin::OpenGlMatrix glMatrix;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            glMatrix.m[i*4 + j] = matrix(j, i); // 注意：按列主序存储
        }
    }
    return glMatrix;
}

voViewer::voViewer(std::string windowName)
{
    windowName_ = windowName;
    
    // Disable X11 shared memory to avoid MESA errors in Docker/remote environments
    setenv("LIBGL_ALWAYS_INDIRECT", "1", 0);
    setenv("MESA_GL_VERSION_OVERRIDE", "3.3", 0);
    
    pangolin::CreateWindowAndBind(windowName, 640,480);
    glEnable(GL_DEPTH_TEST);
    //3D visualizing window
    Eigen::Matrix4f view_point;
    // view_point << 0.987055, -0.0356286, 0.156861, -0.130428, 
    //              -0.0681276, -0.975983, 0.206956, -0.528214, 
    //                0.145719, -0.215005, -0.965752, -2.64094, 
    //                 0, 0, 0, 1; 
    //-- icl的视角
    // view_point << 0.927458, 0.188934, -0.32289, 0.24962, 
    //                0.212732, -0.976329, 0.0396091, -0.171934, 
    //               -0.307765, -0.105475, -0.945667, -1.64501,
    //               0, 0, 0, 1; 
    view_point << 0.927458, 0.188934, -0.32289, 0.197486, 
                  0.212732, -0.976329, 0.0396091, -0.141781, 
                 -0.307765, -0.105475, -0.945667, -1.98199, 
                  0,0,0,1;

    s_cam_ = std::shared_ptr<pangolin::OpenGlRenderState>(
            new pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640,480,420,420,320,240,0.02,500),
                                            view_point));

    handler_ = std::make_shared<pangolin::Handler3D>(*s_cam_);
    d_cam_ = &pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0,-640.0f/480.0f).SetHandler(handler_.get());
    
    //control part image window
    pangolin::CreatePanel("menu").SetBounds(0.3, 1, 0.0, 0.20);
    follow = std::make_shared<pangolin::Var<bool>>("menu.Follow", true, true);
    show_covisibility = std::make_shared<pangolin::Var<bool>>("menu.Co-visibility", false, true);
    slide_bar = std::make_shared<pangolin::Var<double>>("menu.slider", 0.5, 0, 1);
    menuPause = std::make_shared<pangolin::Var<bool>>("menu.Pause", false, true);
    menuShowObjects = std::make_shared<pangolin::Var<bool>>("menu.Show Objects", false, true);
    menuShowLocalEdgeMap = std::make_shared<pangolin::Var<bool>>("menu.Show Local Edge Map", false, true);
    menuShowEnvironment = std::make_shared<pangolin::Var<bool>>("menu.Show Environment Edge Map", false, true);
    menuShowObjectEdgeMap = std::make_shared<pangolin::Var<bool>>("menu.Show Object Edge Map", true, true);
    menuShowAnchorEdgeMap = std::make_shared<pangolin::Var<bool>>("menu.Show Anchor Edge", false, true);
    menuShowSilhouetteEdges = std::make_shared<pangolin::Var<bool>>("menu.Show Silhouette Edges", false, true);
    // New: show raw (current KF) object edges
    menuShowRawObjectEdges = std::make_shared<pangolin::Var<bool>>("menu.Show Raw Object Edges", false, true);

    cameraPose = Eigen::MatrixXd::Identity(4,4);
    gtPose = Eigen::MatrixXd::Identity(4,4);

    pangolin::GetBoundWindow()->RemoveCurrent();
    start();
}

void voViewer::render_loop()
{
    pangolin::BindToContext(windowName_);
    glEnable(GL_DEPTH_TEST);

    while (!pangolin::ShouldQuit() && !stop_)
    {
        // Clear screen first with white background
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pangolin::OpenGlMatrix glMatrix = Eigen2gl(gtPose.cast<float>());

        if(stop_) break;
        if(*follow == true){
            s_cam_->Follow(glMatrix);
        }

        if(stop_) break;
        d_cam_->Activate(*s_cam_);
        
        // Clear again after activating display to ensure background is visible
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //绘制原点标准坐标轴
        Eigen::Matrix4d E = Eigen::MatrixXd::Identity(4,4);
        drawCoordinate(E,0.2);

        //绘制相机位姿
        drawCamera(cameraPose, cv::Vec3b(0,100,255), 0.06);

        //绘制局部地图滑窗
        for(int i = 0; i < sliding_window.size(); ++i)
        {
            drawCamera(sliding_window[i], cv::Vec3b(200,50,50), 0.03);
        }

        if(*menuShowLocalEdgeMap && !localMap_cloud.empty()){
            // If colors are provided, use them; otherwise use default color
            if (!localMap_colors.empty() && localMap_colors.size() == localMap_cloud.size()) {
                for(size_t i = 0; i < localMap_cloud.size(); ++i)
                {
                    drawPointCloudColorSequencial(localMap_cloud[i], localMap_colors[i], 2);
                }
            } else {
                // Fallback to default color if colors not provided
                for(size_t i = 0; i < localMap_cloud.size(); ++i)
                {
                    drawPointCloudColorSequencial(localMap_cloud[i], cv::Vec3b(255, 50, 50), 2);
                }
            }
        }

        if(*menuShowEnvironment && !environment_cloud.empty())
        {
            for(size_t i = 0; i < environment_cloud.size(); ++i)
            {
                const auto& frame = environment_cloud[i];
                // If per-point colors are available and match the points size, use them.
                if (!frame.points.empty() &&
                    frame.points.size() == frame.colors.size()) {
                    drawPointCloudColorful(frame.points, frame.colors, 2);
                } else {
                    // Fallback: draw all points in gray if no colors are provided.
                    drawPointCloudColor(frame.points, cv::Vec3b(150, 150, 150), 2);
                }
            }
        }

        if(*show_covisibility == true)
        {
            drawPointCloudColorful(covisibility_cloud, covisibility_color, 2);
        }

        //绘制轨迹
        if(trajectory.size() > 0){
            drawTrajectory(trajectory, false, cv::Vec3b(0,100,255), 0.05);
            drawTrajectory(trajectory, true, cv::Vec3b(0,100,255), 0.02);
        }

        //绘制真值轨迹
        if(trajectory_GT.size() > 0){
            drawTrajectory(trajectory_GT, false, cv::Vec3b(255,100,0), 0.05);
        }

        // Draw all per-object merged edge clusters as polylines in object color.
        // No silhouette filtering: visualize every cluster regardless of whether it is used in optimization.
        if(*menuShowObjectEdgeMap && map_) {
            const std::vector<dyno::Object*> objects = map_->GetAllObjects();
            for (auto* obj : objects) {
                if (!obj) continue;
                const cv::Scalar c = obj->GetColor();
                // Object color is stored as OpenCV BGR, while VisualizerBase::addPoint/addLine
                // interprets cv::Vec3b as RGB for OpenGL.
                const cv::Vec3b obj_col(static_cast<unsigned char>(c[2]),
                                        static_cast<unsigned char>(c[1]),
                                        static_cast<unsigned char>(c[0]));

                const auto clusters = obj->GetMergedEdgeClusters();
                for (const auto& cloud : clusters) {
                    if (cloud.size() < 2) continue;
                    drawPointCloudColorSequencial(cloud, obj_col, 2);
                }
            }
        }

        // New toggle: draw current per-object raw edges from the most recent KeyFrame.
        if (menuShowRawObjectEdges && *menuShowRawObjectEdges && map_) {
            const std::vector<dyno::KeyFramePtr> kfs = map_->GetAllKeyFrames();
            if (!kfs.empty()) {
                const auto& kf = kfs.back();
                if (kf) {
                    const Eigen::Matrix3d R_wc = kf->KF_pose_g.rotationMatrix();
                    const Eigen::Vector3d t_wc = kf->KF_pose_g.translation();
                    const std::vector<dyno::Object*> objects = map_->GetAllObjects();
                    for (auto* obj : objects) {
                        if (!obj) continue;
                        const int obj_id = static_cast<int>(obj->GetId());
                        const cv::Scalar c = obj->GetColor();
                        const cv::Vec3b obj_col(static_cast<unsigned char>(c[2]),
                                                static_cast<unsigned char>(c[1]),
                                                static_cast<unsigned char>(c[0]));

                        std::vector<cv::Point3d> cloud_world;
                        cloud_world.reserve(2048);
                        for (int i = 0; i < static_cast<int>(kf->mvEdges.size()); ++i) {
                            // Check object association for this edge index
                            bool belongs = false;
                            auto it = kf->mmEdgeIndex2ObjectId.find(i);
                            if (it != kf->mmEdgeIndex2ObjectId.end()) {
                                belongs = (it->second == obj_id);
                            } else if (kf->mvEdges[i].object_id == obj_id) {
                                belongs = true;
                            }
                            if (!belongs) continue;

                            const auto& e = kf->mvEdges[i];
                            for (const auto& pt : e.mvPoints) {
                                if (pt.z_3d <= 0.0) continue;  // skip invalid depth
                                const Eigen::Vector3d pc(pt.x_3d, pt.y_3d, pt.z_3d);
                                const Eigen::Vector3d pw = R_wc * pc + t_wc;
                                cloud_world.emplace_back(pw.x(), pw.y(), pw.z());
                            }
                        }
                        if (cloud_world.size() >= 2) {
                            drawPointCloudColorSequencial(cloud_world, obj_col, 2);
                        }
                    }
                }
            }
        }

        if(*menuShowAnchorEdgeMap && map_) {
            const std::vector<dyno::Object*> objects = map_->GetAllObjects();
            for (auto* obj : objects) {
                if (!obj) continue;
                const cv::Scalar c = obj->GetColor();
                // Convert BGR (OpenCV) -> RGB (OpenGL helper expects RGB ordering).
                const cv::Vec3b obj_col(static_cast<unsigned char>(c[2]),
                                        static_cast<unsigned char>(c[1]),
                                        static_cast<unsigned char>(c[0]));

                const auto anchors = obj->GetAnchorClusters();
                for (const auto& anchor : anchors) {
                    if (anchor.pts.size() < 2) continue;
                    drawPointCloudColorSequencial(anchor.pts, obj_col, 3);
                }
            }
        }

        //绘制地图对象 (ellipsoids)
        if(*menuShowObjects) {
            update_map_objects();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        const Eigen::Matrix4f& modelViewMatrix = s_cam_->GetModelViewMatrix();
        // std::cout<<"*******************************************************"<<std::endl;
        // for (int i = 0; i < 4; ++i) {
        //     for (int j = 0; j < 4; ++j) {
        //         std::cout << modelViewMatrix(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }


        pangolin::FinishFrame();
    }
}

void voViewer::start()
{
    stop_  = false;
    thread = std::thread(std::bind(&voViewer::render_loop, this));
    thread.detach();
}

void voViewer::update_map_objects()
{
    if (!map_) {
        return;  // Map not set yet
    }

    // Get all objects from the shared EdgeMap
    std::vector<dyno::Object*> objects = map_->GetAllObjects();

    glPointSize(1);
    glLineWidth(2);

    for (auto* obj : objects) {
        if (!obj || obj->isBad()) continue;

        // Use the per-object color
        cv::Scalar c = obj->GetColor();
        glColor3f(static_cast<double>(c(2)) / 255.0,
                  static_cast<double>(c(1)) / 255.0,
                  static_cast<double>(c(0)) / 255.0);

        // Draw the ellipsoid as a set of line strips
        const dyno::Ellipsoid& ell = obj->GetEllipsoid();
        auto pts = ell.GeneratePointCloud();
        int i = 0;
        while (i < pts.rows()) {
            glBegin(GL_LINE_STRIP);
            for (int k = 0; k < 50 && i < pts.rows(); ++k, ++i) {
                glVertex3f(pts(i, 0), pts(i, 1), pts(i, 2));
            }
            glEnd();
        }
    }
}
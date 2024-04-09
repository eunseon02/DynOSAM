
table = triangulation1;

cam_pose_origin = table2array(table(1:4, :));
cam_pose_target = table2array(table(6:9, :));

points_origin = table2array(table(11:22, 1:3));
points_target = table2array(table(24:35, 1:3));
points_est = table2array(table(37:48, 1:3));

x_axis = [1; 0; 0; 1];
y_axis = [0; 1; 0; 1];
z_axis = [0; 0; 1; 1];

x_origin = cam_pose_origin*x_axis;
y_origin = cam_pose_origin*y_axis;
z_origin = cam_pose_origin*z_axis;
o_origin = cam_pose_origin(1:3, 4);

x_target = cam_pose_target*x_axis;
y_target = cam_pose_target*y_axis;
z_target = cam_pose_target*z_axis;
o_target = cam_pose_target(1:3, 4);

axis_linewidth = 3;
point_size = 20;

figure();
plot3(points_origin(:, 1), points_origin(:, 2), points_origin(:, 3), "g.", "MarkerSize", point_size);
hold on;
plot3(points_target(:, 1), points_target(:, 2), points_target(:, 3), "b.", "MarkerSize", point_size);
plot3(points_est(:, 1), points_est(:, 2), points_est(:, 3), "r.", "MarkerSize", point_size);

plot3([o_origin(1), x_origin(1)], [o_origin(2), x_origin(2)], [o_origin(3), x_origin(3)], "r-", "LineWidth", axis_linewidth);
plot3([o_origin(1), y_origin(1)], [o_origin(2), y_origin(2)], [o_origin(3), y_origin(3)], "g-", "LineWidth", axis_linewidth);
plot3([o_origin(1), z_origin(1)], [o_origin(2), z_origin(2)], [o_origin(3), z_origin(3)], "b-", "LineWidth", axis_linewidth);

plot3([o_target(1), x_target(1)], [o_target(2), x_target(2)], [o_target(3), x_target(3)], "r-", "LineWidth", axis_linewidth);
plot3([o_target(1), y_target(1)], [o_target(2), y_target(2)], [o_target(3), y_target(3)], "g-", "LineWidth", axis_linewidth);
plot3([o_target(1), z_target(1)], [o_target(2), z_target(2)], [o_target(3), z_target(3)], "b-", "LineWidth", axis_linewidth);

hold off;
grid on;
axis equal;

xlim([-2, 22]);
ylim([-11, 11]);
zlim([-2, 22]);

xlabel("X", "FontSize", 16);
ylabel("Y", "FontSize", 16);
zlabel("Z", "FontSize", 16);

title("Rotation-Compensated Triangulation with Motion", "FontSize", 20);
legend("Points at step 0", "Points at step 1", "Estimated points at step 0", "Location","northeast", "Fontsize", 18);

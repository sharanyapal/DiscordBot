// Import required modules
const cv2 = require('cv2');
const np = require('numpy');
const math = require('math');
const { get_face_detector, find_faces } = require('./face_detector.js');
const { get_landmark_model, detect_marks, draw_marks } = require('./face_landmarks.js');

function get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val) {
    // Return the 3D points present as 2D for making annotation box
    let point_3d = [];
    let dist_coeffs = np.zeros([4, 1]);
    let rear_size = val[0];
    let rear_depth = val[1];
    point_3d.push([-rear_size, -rear_size, rear_depth]);
    point_3d.push([-rear_size, rear_size, rear_depth]);
    point_3d.push([rear_size, rear_size, rear_depth]);
    point_3d.push([rear_size, -rear_size, rear_depth]);
    point_3d.push([-rear_size, -rear_size, rear_depth]);

    let front_size = val[2];
    let front_depth = val[3];
    point_3d.push([-front_size, -front_size, front_depth]);
    point_3d.push([-front_size, front_size, front_depth]);
    point_3d.push([front_size, front_size, front_depth]);
    point_3d.push([front_size, -front_size, front_depth]);
    point_3d.push([-front_size, -front_size, front_depth]);
    point_3d = np.array(point_3d, np.float).reshape([-1, 3]);

    // Map to 2d img points
    let point_2d = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs);
    point_2d = np.int32(point_2d.reshape([-1, 2]));
    return point_2d;
}

function draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, rear_size = 300, rear_depth = 0, front_size = 500, front_depth = 400, color = [255, 255, 0], line_width = 2) {
    // Draw a 3D anotation box on the face for head pose estimation
    rear_size = 1;
    rear_depth = 0;
    front_size = img.shape[1];
    front_depth = front_size*2;
    let val = [rear_size, rear_depth, front_size, front_depth];
    let point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val);

    // Draw all the lines
    cv2.polylines(img, [point_2d], true, color, line_width, cv2.LINE_AA);
    cv2.line(img, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA);
    cv2.line(img, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA);
    cv2.line(img, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA);
}

function head_pose_points(img, rotation_vector, translation_vector, camera_matrix) {
    // Get the points to estimate head pose sideways
    let rear_size = 1;
    let rear_depth = 0;
    let front_size = img.shape[1];
    let front_depth = front_size*2;
    let val = [rear_size, rear_depth, front_size, front_depth];
    let point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val);
    let y = (point_2d[5] + point_2d[8]) / 2;
    let x = point_2d[2];

    return [x, y];

    let face_model = get_face_detector();
    let landmark_model = get_landmark_model();
    let outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]];
    let d_outer = Array(5).fill(0);
    let inner_points = [[61, 67], [62, 66], [63, 65]];
    let d_inner = Array(3).fill(0);
    let font = cv2.FONT_HERSHEY_SIMPLEX;
    let cap = cv2.VideoCapture(0);

    let ret, img = cap.read();
    let size = img.shape;

    let model_points = [    [0.0, 0.0, 0.0],             // Nose tip
                            [0.0, -330.0, -65.0],        // Chin
                            [-225.0, 170.0, -135.0],     // Left eye left corner
                            [225.0, 170.0, -135.0],      // Right eye right corne
                            [-150.0, -150.0, -125.0],    // Left Mouth corner
                            [150.0, -150.0, -125.0]      // Right mouth corner
    ];

    let focal_length = size[1];
    let center = [size[1] / 2, size[0] / 2];
    let camera_matrix = [    [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]

    while (true) {
        let ret, img = cap.read();
        let rects = find_faces(img, face_model);
        for (let rect of rects) {
            let shape = detect_marks(img, landmark_model, rect);
            draw_marks(img, shape);
            cv2.putText(img, 'Press r to record Mouth distances', [30, 30], font, 1, [0, 255, 255], 2);
            cv2.imshow('Output', img);
        }
        for (let i = 0; i < 100; i++) {
            for (let i = 0; i < outer_points.length; i++) {
                let p1 = outer_points[i][0], p2 = outer_points[i][1];
                d_outer[i] += shape[p2][1] - shape[p1][1];
            }
            for (let i = 0; i < inner_points.length; i++) {
                let p1 = inner_points[i][0], p2 = inner_points[i][1];
                d_inner[i] += shape[p2][1] - shape[p1][1];
            }
        }
        break;
    }
    cv2.destroyAllWindows();
    d_outer = d_outer.map(x => x / 100);
    d_inner = d_inner.map(x => x / 100);

    while (true) {
        let ret, img = cap.read();
        let rects = find_faces(img, face_model);
        for (let rect of rects) {
            let shape = detect_marks(img, landmark_model, rect);
            let cnt_outer = 0;
            let cnt_inner = 0;
            draw_marks(img, shape.slice(48));
            for (let i = 0; i < outer_points.length; i++) {
                let [p1, p2] = outer_points[i];
                if (d_outer[i] + 3 < shape[p2][1] - shape[p1][1]) {
                    cnt_outer++;
                }
            }
            for (let i = 0; i < inner_points.length; i++) {
                let [p1, p2] = inner_points[i];
                if (d_inner[i] + 2 < shape[p2][1] - shape[p1][1]) {
                    cnt_inner++;
                }
            }
            if (cnt_outer > 3 && cnt_inner > 2) {
                console.log('Mouth open');
                cv2.putText(img, 'Mouth open', [30, 30], font, 1, [0, 255, 255], 2);
            }
        }
        if (ret == true) {
            let faces = find_faces(img, face_model);
            for (let face of faces) {
                let marks = detect_marks(img, landmark_model, face);
                let image_points = [
                                    marks[30], // Nose tip
                                    marks[8], // Chin
                                    marks[36], // Left eye left corner
                                    marks[45], // Right eye right corner
                                    marks[48], // Left mouth corner
                                    marks[54] // Right mouth corner
                ];
                let dist_coeffs = np.zeros([4,1]); // Assuming no lens distortion
                let { success, rotation_vector, translation_vector } = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, cv2.SOLVEPNP_UPNP);
                // Project a 3D point (0, 0, 1000.0) onto the image plane.
                // We use this to draw a line sticking out of the nose
                let [nose_end_point2D, jacobian] = cv2.projectPoints([0, 0, 1000], rotation_vector, translation_vector, camera_matrix, dist_coeffs);
                for (let p of image_points) {
                    cv2.circle(img, [Math.round(p[0]), Math.round(p[1])], 3, [0,0,255], -1);
                }
                let p1 = [Math.round(image_points[0][0]), Math.round(image_points[0][1])];
                let p2 = [Math.round(nose_end_point2D[0][0][0]), Math.round(nose_end_point2D[0][0][1])];
                let [x1, x2] = head_pose_points(img, rotation_vector, translation_vector, camera_matrix);
                cv2.line(img, p1, p2, [0, 255, 255], 2);
                cv2.line(img, [Math.round(x1[0]), Math.round(x1[1])], [Math.round(x2[0]), Math.round(x2[1])], [255, 255, 0], 2);
                try {
                    let m = (p2[1] - p1[1])/(p2[0] - p1[0]);
                    let ang1 = parseInt(Math.atan(m)*180/Math.PI);
                } catch (e) {
                    let ang1 = 90;
                }
                    
                try {
                    let m = (x2[1] - x1[1])/(x2[0] - x1[0]);
                    let ang2 = parseInt(Math.atan(-1/m)*180/Math.PI);
                } catch (e) {
                    let ang2 = 90;
                }
                    
                if (ang1 >= 48) {
                    console.log('Head down');
                    cv2.putText(img, 'Head down', [30, 30], font, 2, [255, 255, 128], 3);
                } else if (ang1 <= -48) {
                    console.log('Head up');
                    cv2.putText(img, 'Head up', [30, 30], font, 2, [255, 255, 128], 3);
                }
                    
                if (ang2 >= 48) {
                    console.log('Head right');
                    cv2.putText(img, 'Head right', [90, 30], font, 2, [255, 255, 128], 3);
                } else if (ang2 <= -48) {
                    console.log('Head left');
                    cv2.putText(img, 'Head left', [90, 30], font, 2, [255, 255, 128], 3);
                }
                    
                cv2.putText(img, ang1.toString(), p1, font, 2, [128, 255, 255], 3);
                cv2.putText(img, ang2.toString(), x1, font, 2, [255, 255, 128], 3);
            }
            cv2.imshow('img', img);
        }                
        if (cv2.waitKey(1) & 0xFF === 'q'.charCodeAt(0)) {
            process.exit(0);
        } 
    } 
}
cv2.destroyAllWindows();
cap.release();
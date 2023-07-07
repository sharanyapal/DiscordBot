let cv2 = require('opencv4nodejs');
let np = require('numpy');
let { get_face_detector, find_faces } = require('./face_detector');
let { get_landmark_model, detect_marks } = require('./face_landmarks');

function eye_on_mask(mask, side, shape) {
    let points = side.map(i => shape[i]);
    points = np.array(points, np.int32);
    mask = cv2.fillConvexPoly(mask, points, 255);
    let l = points[0][0];
    let t = Math.floor((points[1][1] + points[2][1]) / 2);
    let r = points[3][0];
    let b = Math.floor((points[4][1] + points[5][1]) / 2);
    return [mask, [l, t, r, b]];
}

function find_eyeball_position(end_points, cx, cy) {
    let x_ratio = (end_points[0] - cx) / (cx - end_points[2]);
    let y_ratio = (cy - end_points[1]) / (end_points[3] - cy);
    if (x_ratio > 3) {
        return 1;
    } else if (x_ratio < 0.33) {
        return 2;
    } else if (y_ratio < 0.33) {
        return 3;
    } else {
        return 0;
    }
}

function contouring(thresh, mid, img, end_points, right = false) {
    let cnts = thresh.findContours(cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
    try {
        let cnt = cnts.reduce((acc, val) => acc.area > val.area ? acc : val);
        let M = cnt.moments();
        let cx = Math.floor(M['m10'] / M['m00']);
        let cy = Math.floor(M['m01'] / M['m00']);
        if (right) {
            cx += mid;
        }
        cv2.circle(img, new cv2.Point2(cx, cy), 4, new cv2.Scalar(0, 0, 255), 2);
        let pos = find_eyeball_position(end_points, cx, cy);
        return pos;
    } catch (error) {
        console.error(error);
    }
}

function process_thresh(thresh) {
    thresh = cv2.erode(thresh, new cv2.Mat(), { iterations: 2 });
    thresh = cv2.dilate(thresh, new cv2.Mat(), { iterations: 4 });
    thresh = cv2.medianBlur(thresh, 3);
    thresh = cv2.bitwise_not(thresh);
    return thresh;
}

function print_eye_pos(img, left, right) {
    if (left == right && left != 0) {
        let text = '';
        if (left == 1) {
            console.log('Looking left');
            text = 'Looking left';
        } else if (left == 2) {
            console.log('Looking right');
            text = 'Looking right';
        } else if (left == 3) {
            console.log('Looking up');
            text = 'Looking up';
        }
        let font = cv2.FONT_HERSHEY_SIMPLEX;
        cv2.putText(img, text, new cv2.Point2(30, 30), font, 1, new cv2.Scalar(0, 255, 255), 2, cv2.LINE_AA);
    }
}

let face_model = get_face_detector();
let landmark_model = get_landmark_model();
let left = [36, 37, 38, 39, 40, 41];
let right = [42, 43, 44, 45, 46, 47];

let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');

let kernel = cv2.getStructuringElement(cv2.MORPH_RECT, new cv2.Size(9, 9));

function onThresholdChange(event) {
    let threshold = event.target.value;
    cv2.imshow('threshold', threshold);
}

cv2.createTrackbar('threshold', 'image', 75, 255, onThresholdChange);

async function run() {
    let stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
}

while(true) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    let img = cv2.imread(canvas);

    let rects = find_faces(img, face_model);

    for (let rect of rects) {
        let shape = detect_marks(img, landmark_model, rect);
        let mask = new cv2.Mat.zeros(img.rows, img.cols, cv2.CV_8UC1);
        mask = eye_on_mask(mask, left, shape);
        mask = eye_on_mask(mask, right, shape);
        mask = cv2.dilate(mask, kernel, 5);

        let eyes = new cv2.Mat.zeros(img.rows, img.cols, cv2.CV_8UC3);
        cv2.bitwise_and(img, img, eyes, mask);

        let maskData = eyes.data8().every((val, i) => val === 0);
        eyes.data8().forEach((val, i) => {
                if (maskData[i]) {
                    eyes.data8()[i] = 255;
                }
            };

            let mid = Math.floor((shape[42].x + shape[39].x) / 2);
            let eyesGray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY);
            let threshold = cv2.getTrackbarPos('threshold', 'image');
            let thresh = new cv2.Mat.zeros(img.rows, img.cols, cv2.CV_8UC1); cv2.threshold(eyesGray, thresh, threshold, 255, cv2.THRESH_BINARY); thresh = process_thresh(thresh);

            let eyeballPosLeft = contouring(thresh, mid, img, end_points_left);
            let eyeballPosRight = contouring(thresh, mid, img, end_points_right, true); print_eye_pos(img, eyeballPosLeft, eyeballPosRight);
        }

        cv2.imshow()
        cv2.imshow()

        if (cv2.waitKey(1) & 0xFF === 'q'.charCodeAt(0)) {
            process.exit(0);
        } 
    }
}
cv2.destroyAllWindows();
cap.release();
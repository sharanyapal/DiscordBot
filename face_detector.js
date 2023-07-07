function get_face_detector(modelFile=null, configFile=null, quantized=false) {
    let model;
    if (quantized) {
        if (modelFile === null) {
            modelFile = 'models/opencv_face_detector_uint8.pb';
        }
        if (configFile === null) {
            configFile = 'models/opencv_face_detector.pbtxt';
        }
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile);
    } else {
        if (modelFile === null) {
            modelFile = 'models/res10_300x300_ssd_iter_140000.caffemodel';
        }
        if (configFile === null) {
            configFile = 'models/deploy.prototxt';
        }
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile);
    }
    return model;
}

function find_faces(img, model) {
    const [h, w] = [img.shape[0], img.shape[1]];
    const blob = cv2.dnn.blobFromImage(cv2.resize(img, [300, 300]), 1.0, [300, 300], [104.0, 177.0, 123.0]);
    model.setInput(blob);
    const res = model.forward();
    const faces = [];
    for (let i = 0; i < res.shape[2]; i++) {
        const confidence = res[0, 0, i, 2];
        if (confidence > 0.5) {
            const box = res[0, 0, i, 3:7] * np.array([w, h, w, h]);
            const [x, y, x1, y1] = box.astype('int');
            faces.push([x, y, x1, y1]);
        }
    }
    return faces;
}

function draw_faces(img, faces) {
    for (const [x, y, x1, y1] of faces) {
        cv2.rectangle(img, [x, y], [x1, y1], [0, 0, 255], 3);
    }
}
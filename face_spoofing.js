const cv = require('opencv4nodejs');
const { get_face_detector, find_faces } = require('./face_detector');
const joblib = require('joblib');
const np = require('numpy');

function calc_hist(img) {
  /**
   * To calculate histogram of an RGB image
   *
   * Parameters
   * ----------
   * img : Array of uint8
   *     Image whose histogram is to be calculated
   *
   * Returns
   * -------
   * histogram : np.array
   *     The required histogram
   */
  let histogram = [0, 0, 0];
  for (let j = 0; j < 3; j++) {
    const histr = cv.calcHist([img], [j], null, [256], [0, 256]);
    histr.mul(255.0 / histr.max());
    histogram[j] = histr;
  }
  return np.array(histogram);
}

const face_model = get_face_detector();
const clf = joblib.load('models/face_spoofing.pkl');
const cap = new cv.VideoCapture(0);

const sample_number = 1;
let count = 0;
const measures = np.zeros(sample_number, np.float32);

while (true) {
  const img = cap.read();
  const faces = find_faces(img, face_model);

  measures.put(count % sample_number, 0);
  const { rows: height, cols: width } = img.sizes;
  for (const face of faces) {
    const [x, y, x1, y1] = face;
    const roi = img.getRegion(new cv.Rect(x, y, x1 - x, y1 - y));
    const point = new cv.Point2(x, y - 5);

    const img_ycrcb = roi.cvtColor(cv.COLOR_BGR2YCrCb);
    const img_luv = roi.cvtColor(cv.COLOR_BGR2Luv);

    const ycrcb_hist = calc_hist(img_ycrcb);
    const luv_hist = calc_hist(img_luv);

    let feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel());
    feature_vector = feature_vector.reshape(1, feature_vector.length);

    const prediction = clf.predict_proba(feature_vector);
    const prob = prediction.get(0, 1);

    measures.put(count % sample_number, prob);

    img.drawRectangle(new cv.Rect(x, y, x1 - x, y1 - y), new cv.Vec(255, 0, 0), 2);

    // console.log(measures, np.mean(measures))
    if (measures.findIndex((x) => x === 0) < 0) {
      let text = 'True';
      if (np.mean(measures) >= 0.7) {
        text = 'False';
        img.putText(
          text,
          point,
          cv.FONT_HERSHEY_SIMPLEX,
          0.9,
          new cv.Vec(0, 0, 255),
          2,
          cv.LINE_AA
        );
      } else {
        img.putText(
          text,
          point,
          cv.FONT_HERSHEY_SIMPLEX,
          0.9,
          new cv.Vec(0, 255, 0),
          2,
          cv.LINE_AA
        );
      }
    }
  }
  count++;
  cv.imshowWait('img_rgb', img);

  if (cv.waitKey(1) & 0xff === 'q'.charCodeAt(0)) {
    break;
  }
}

cap.release();
cv.destroyAllWindows();

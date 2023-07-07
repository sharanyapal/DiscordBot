const cv = require('opencv');
const tf = require('@tensorflow/tfjs-node');
const { promisify } = require('util');
const fs = require('fs');

const readFileAsync = promisify(fs.readFile);

async function get_landmark_model(saved_model='models/pose_model') {
    const buffer = await readFileAsync(saved_model);
    const model = await tf.node.loadSavedModel(buffer);
    return model;
}

function get_square_box(box) {
    const left_x = box[0];
    const top_y = box[1];
    const right_x = box[2];
    const bottom_y = box[3];

    const box_width = right_x - left_x;
    const box_height = bottom_y - top_y;

    // Check if box is already a square. If not, make it a square.
    const diff = box_height - box_width;
    const delta = parseInt(Math.abs(diff) / 2);

    if (diff === 0) {                   // Already a square.
        return box;
    } else if (diff > 0) {              // Height > width, a slim box.
        const new_left_x = left_x - delta;
        const new_right_x = right_x + delta;
        const new_bottom_y = bottom_y + ((diff % 2) === 1 ? 1 : 0);

        return [new_left_x, top_y, new_right_x, new_bottom_y];
    } else {                           // Width > height, a short box.
        const new_top_y = top_y - delta;
        const new_bottom_y = bottom_y + delta;
        const new_right_x = right_x + ((diff % 2) === 1 ? 1 : 0);

        return [left_x, new_top_y, new_right_x, new_bottom_y];
    }
}

function move_box(box, offset) {
    const left_x = box[0] + offset[0];
    const top_y = box[1] + offset[1];
    const right_x = box[2] + offset[0];
    const bottom_y = box[3] + offset[1];
    return [left_x, top_y, right_x, bottom_y];
}

function detectMarks(img, model, face) {
  /*
   * Find the facial landmarks in an image from the faces
   *
   * Parameters
   * ----------
   * img : np.uint8
   *   The image in which landmarks are to be found
   * model : Tensorflow model
   *   Loaded facial landmark model
   * face : list
   *   Face coordinates (x, y, x1, y1) in which the landmarks are to be found
   *
   * Returns
   * -------
   * marks : numpy array
   *   facial landmark points
   */

  const offset_y = parseInt(Math.abs((face[3] - face[1]) * 0.1));
  const box_moved = moveBox(face, [0, offset_y]);
  const facebox = getSquareBox(box_moved);

  const [h, w] = img.shape.slice(0, 2);
  if (facebox[0] < 0) facebox[0] = 0;
  if (facebox[1] < 0) facebox[1] = 0;
  if (facebox[2] > w) facebox[2] = w;
  if (facebox[3] > h) facebox[3] = h;

  const face_img = img.slice(
    facebox[1], facebox[3], facebox[0], facebox[2]
  );
  const resized_img = cv2.resize(face_img, [128, 128]);
  const rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB);

  // Actual detection.
  const predictions = model.signatures["predict"](
    tf.constant([rgb_img], dtype=tf.uint8)
  );

  // Convert predictions to landmarks.
  let marks = np.array(predictions['output']).flatten().slice(0, 136);
  marks = np.reshape(marks, [-1, 2]);

  marks *= (facebox[2] - facebox[0]);
  marks[:, 0] += facebox[0];
  marks[:, 1] += facebox[1];
  marks = marks.astype(np.uint);

  return marks;
}

function drawMarks(image, marks, color=[0, 255, 0]) {
  /*
   * Draw the facial landmarks on an image
   *
   * Parameters
   * ----------
   * image : np.uint8
   *   Image on which landmarks are to be drawn.
   * marks : list or numpy array
   *   Facial landmark points
   * color : tuple, optional
   *   Color to which landmarks are to be drawn with. The default is [0, 255, 0].
   *
   * Returns
   * -------
   * None.
   */

  for (const mark of marks) {
    cv2.circle(image, [mark[0], mark[1]], 2, color, -1, cv2.LINE_AA);
  }
}
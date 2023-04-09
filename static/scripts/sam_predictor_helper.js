
// create a new session and load the specific model.
const sam_onnx_model_path = "./static/sam_onnx_quantized_example.onnx"
const ort_session = await ort.InferenceSession.create(sam_onnx_model_path);

// Default inputs
const mask_input = new Float32Array(256*256).fill(0);
const mask_inputT = new ort.Tensor('float32', mask_input, [1, 1, 256, 256]);

const has_mask_input = new Float32Array(1).fill(0);
const has_mask_inputT = new ort.Tensor('float32', has_mask_input, [1]);

// Funcs to set image info
var image_embeddingT;
export function set_image_embedding(image_embedding) {
    image_embeddingT = new ort.Tensor('float32', image_embedding, [1, 256, 64, 64])
}

var targ_im_scale;
var orig_im_sizeT;
export function set_image_size(img_h, img_w) {
    targ_im_scale = get_targ_scale(img_h, img_w);

    const orig_im_size = Float32Array.from([img_h, img_w]);
    orig_im_sizeT = new ort.Tensor('float32', orig_im_size, [2]);
}


// Scaling points helpers
const SAM_DEFAULT_IMG_SIZE = 1024

function get_targ_scale(img_h, img_w) {
    return SAM_DEFAULT_IMG_SIZE * 1.0 / Math.max(img_h, img_w);
}

function rescale_coords(coords) {
    // Copy
    coords = JSON.parse(JSON.stringify(coords));
    // Apply scaling
    coords.forEach(coord => {
        coord[0] = coord[0] * targ_im_scale;
        coord[1] = coord[1] * targ_im_scale;
    });
    return coords;
}

// Run ONNX model
// use an async context to call onnxruntime functions.
export async function run_sam(points, neg_points) {
    

    let extra_zero = [[0.0, 0.0]];

    var all_points;
    if (neg_points) {
        all_points = points.concat(neg_points, extra_zero); // add 0,0
    } else {
        all_points = points.concat(extra_zero);
    }
    const all_points_scaled = rescale_coords(all_points);

    const n_points = all_points_scaled.length;
    const labels = new Array(n_points).fill(1);
    labels.fill(-1, points.length);

    // prepare inputs. a tensor need its corresponding TypedArray as data
    const point_coords = Float32Array.from(all_points_scaled.flat(1));
    const point_coordsT = new ort.Tensor('float32', point_coords, [1, n_points, 2]);

    const point_labels = Float32Array.from(labels);
    const point_labelsT = new ort.Tensor('float32', point_labels, [1, n_points]);

    // prepare feeds. use model input names as keys.
    const feeds = { 'image_embeddings': image_embeddingT,
                    'point_coords': point_coordsT,
                    'point_labels': point_labelsT,
                    'mask_input': mask_inputT,
                    'has_mask_input': has_mask_inputT,
                    'orig_im_size': orig_im_sizeT };

    // feed inputs and run
    const results = await ort_session.run(feeds);
    return results.masks.data;
}
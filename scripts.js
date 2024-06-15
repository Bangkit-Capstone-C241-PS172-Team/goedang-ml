async function loadModel() {
    const model = await tf.loadLayersModel('../models/tfjs_model/model.json');
    return model;
}

function createWindowedDataset(series, windowSize) {
    const windows = [];
    for (let i = 0; i < series.length - windowSize + 1; i++) {
        const window = series.slice(i, i + windowSize);
        windows.push(window);
    }
    return windows;
}

async function predict() {
    const model = await loadModel();
    const windowSize = 10;

    // Contoh input data, sesuaikan dengan input yang sesuai dengan modelmu
    const data = [
        [0.30765789], [0.31075658], [0.31384868], [0.31694737], [0.32004605], [0.32313816], [0.32623684], [0.32932895],
        [0.33242763], [0.33552632], [0.33552632], [0.33552632], [0.33552632], [0.33917763], [0.34283553], [0.34648684],
        [0.35014474], [0.35379605], [0.35745395], [0.36110526], [0.36476316], [0.36842105], [0.36842105], [0.36842105],
        [0.36653947], [0.36465789], [0.36277632], [0.36090132], [0.35901974], [0.35713816], [0.35526316], [0.35526316],
        [0.35526316], [0.75657895], [0.66315789], [0.56973684], [0.47631579], [0.38289474], [0.28947368], [0.31359211],
        [0.33771711], [0.36184211], [0.38596053], [0.41008553], [0.43421053], [0.45832895], [0.48245395], [0.50657895],
        [0.19736842], [0.19736842], [0.19736842], [0.19736842], [0.19407895], [0.19078947], [0.1875], [0.18421053],
        [0.18092105], [0.17763158], [0.17434211], [0.17105263], [0.16776316], [0.16447368], [0.16118421], [0.15789474],
        [0.15460526], [0.15131579], [0.14802632], [0.14473684], [0.14144737], [0.13815789], [0.13486842], [0.13157895],
        [0.12828947], [0.125], [0.125], [0.125], [0.12029605], [0.11559868], [0.11090132], [0.10619737], [0.1015],
        [0.09680263], [0.09210526], [0.09210526], [0.20394737], [0.31578947], [0.74342105], [0.31578947], [0.31578947],
        [0.31578947], [0.31578947], [0.27631579], [0.63815789], [1.0], [0.25], [0.32236842], [0.39473684], [0.46710526],
        [0.53947368], [0.61184211], [0.68421053], [0.61513158], [0.54605263], [0.40131579], [0.25657895], [0.11184211]
    ];

    if (data.length < windowSize) {
        alert('Input data kurang dari window size');
        return;
    }

    // Membuat windowed dataset
    const windows = createWindowedDataset(data, windowSize);

    // Mengambil window terakhir untuk prediksi
    const lastWindow = windows[windows.length - 1];
    const inputTensor = tf.tensor2d(lastWindow, [1, windowSize, 1]);

    // Lakukan prediksi
    const prediction = model.predict(inputTensor);
    const predictionArray = await prediction.array();

    // Tampilkan hasil prediksi dalam alert
    alert(`Prediction: ${predictionArray[0][0]}`);
}

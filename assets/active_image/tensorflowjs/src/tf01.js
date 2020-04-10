// df: cleaned dataset
let df_total,df,model,tensorData;
// load and clean data, trigger when input changes
function readFile(input) {
    let file = input.files[0];
    let reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function() {
        df_total = Papa.parse(reader.result, {delimiter: ",",newline: "\n",header: true,}).data;
        df = df_total.map(car => ({
            mpg: Number(car.mpg),
            horsepower: Number(car.horsepower),
        })).filter(car => (!isNaN(car.mpg) && !isNaN(car.horsepower)));
    };
    reader.onerror = function() {
        console.log(reader.error);
    };
    updateStatus("Data loaded.");
}

function dataPreview(){
    const headers = [];
    for (let i in df_total[0]) {
        headers.push(i);
    }
    let previewData = df_total.slice(0,10).map(d => {
        let row = [];
        for (let i in d) {
            row.push(d[i]);
        }
        return row;
    });
    tfvis.render.table(
        {name: 'data', tab: 'data preview'},
        { headers, values:previewData }
        );
    updateStatus(`Shape of the data: (${df_total.length},${headers.length})`);
    updateStatus("First 10 rows are shown.");
}

function updateStatus(text) {
    let para=document.createElement("p");
    let node=document.createTextNode(text);
    para.appendChild(node);
    document.getElementById("status").appendChild(para);
}

function scatterPlot() {
    const values = df.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));
    tfvis.render.scatterplot(
        {name: 'Horsepower vs MPG', tab: 'scatter plot'},
        {values: [values],series: ["original"]},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 400
        }
    );
    updateStatus("Scatter plot rendered.");
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();
    // simple linear model
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
}


function modelSummary(){
    model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary', tab: 'summary'}, model);
    updateStatus("Model initiated.");
}

function convertToTensor(df) {
    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(df);

        // Step 2. Convert Array to Tensor
        const inputs = df.map(d => d.horsepower);
        const labels = df.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

async function compileAndFit(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    updateStatus("Training model...");
    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'training' },
            ['mse'],
            { height: 400, callbacks: ['onEpochEnd'] }
        )
    });
}

async function trainModel() {
    // Convert the data to a form we can use for training.
    tensorData = convertToTensor(df);
    const {inputs, labels} = tensorData;

    // Train the model
    await compileAndFit(model, inputs, labels);
    updateStatus("Training finished.");
}


function makePrediction(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    const [xhat, yhat] = tf.tidy(() => {

        const xhat0 = tf.linspace(0, 1, 100);
        const yhat0 = model.predict(xhat0.reshape([100, 1]));

        const xhat = xhat0
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const yhat = yhat0
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // Un-normalize the data
        return [xhat.dataSync(), yhat.dataSync()];
    });


    const predictedPoints = Array.from(xhat).map(
        (val, i) => ({
                x: val, y: yhat[i]}
    ));

    const originalPoints = inputData.map(
        d => ({
            x: d.horsepower, y: d.mpg,
    }));


    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data', tab: 'prediction plot'},
        {values: [originalPoints, predictedPoints], series: ['original','predicted']},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 400
        }
    );
}

function predictionPlot(){
    makePrediction(model,df,tensorData)
    updateStatus("Model prediction is plotted.");
}







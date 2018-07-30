async function fitModel(model, xs, ys, validationData) {
  console.log(xs.length)
  const fitParams = {
       batchSize: xs.size[0],
       validationData,
       epochs: 1
   }
  return model.fit(xs, xs, fitParams)
}

async function plotLoss(values, type='loss') {
  var ctx = document.getElementById(`${type}Chart`).getContext("2d");

  var myChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: [...values.keys()],
          datasets: [{
              label: type,
              data: values,
          }]
      }
  });
}



class Dataset {
  constructor (trainData, testData={xs:[], ys:[]}, batchSize=3, shuffle=true) {
    this.trainData = trainData
    this.testData = testData
    this.batchSize = batchSize
    this.shuffle = shuffle

    this.reset()
    this.getTrainBatch = this.generateBatchFunction('trainData')
    this.getTestBatch = this.generateBatchFunction('testData')
  }

  generateBatchFunction (dataType) {
    const getBatch = (batchSize=this.batchSize, shuffle = this.shuffle) => {
      const batchStart = this[`_${dataType}_batchStart`];
      const batchEnd = batchStart + batchSize;
      //update batch starting position
      console.log('xss', this[dataType].xs);
      console.log('size', this[dataType].xs.size);
      console.log('shape', this[dataType].xs.shape);
      (batchEnd >= this[dataType].xs.shape[0]) ? this.reset(dataType) : this[`_${dataType}_batchStart`] = batchEnd;
      if (batchEnd >= this[dataType].xs.shape[0]) {
        var result =  {xs: this[dataType].xs.slice(batchStart, -1), ys:this[dataType].ys.slice(batchStart, -1)} //slice until the end of the tensor
        this.reset(dataType)
      } else {
        this[`_${dataType}_batchStart`] = batchEnd;
        var result = {xs: this[dataType].xs.slice(batchStart, batchSize), ys:this[dataType].ys.slice(batchStart, batchSize)}
      }
      return result
    }
    return getBatch
  } 

  reset(dataType='both', shuffle=this.shuffle) {
    // resets batch starting point for trainData, testData or both
    // and shuffles the data if shuffle=true
    const dataTypes = dataType == 'both' ? ['trainData', 'testData'] : [dataType]
    dataTypes.forEach((dataType) => {
      this[`_${dataType}_batchStart`] = 0  // reset starting point for next batch
      if (shuffle) { this[dataType].sort(() => Math.random() - 0.5 )} //shuffle data
        //TODDO FIX TO WORK WITH X AND Y ^
    })
  }
}


function generateQuickGuid() {
    return Math.random().toString(36).substring(2, 15) +
        Math.random().toString(36).substring(2, 15);
}

function loadFile(filePath) {
  var result = null;
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.open("GET", filePath, false);
  xmlhttp.send();
  if (xmlhttp.status==200) {
    result = xmlhttp.responseText;
  }
  return result;
}

function displayMarkDown(url) {
  var fileText = loadFile(url)
    converter = new showdown.Converter(),
    html      = converter.makeHtml(fileText);
    document.getElementById("blog-post").innerHTML = html
    return html
}
// Setting Intrepeter
val tfliteOptions = Interpreter.Options()
tfliteOptions.setNumThreads(5)
tfliteOptions.setUseNNAPI(true)

// Load Model to Intrepeter
assetManager.openFd("workout_model.tflite")
val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
val fileChannel = inputStream.channel
val startOffset = fileDescriptor.startOffset
val declaredLength = fileDescriptor.declaredLength
tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

// Load Label
 labelList = assetManager.open("workout_label.txt")
    .bufferedReader()
    .useLines { it.toList() }

// Initializing Intrepeter
tflite = Interpreter(tfliteModel, tfliteOptions)

// Preparing Input
byteBuffer.put((intValue.shr(16) and 0xFF).toByte())
byteBuffer.put((intValue.shr(8) and 0xFF).toByte())
byteBuffer.put((intValue and 0xFF).toByte())

// Running Inference
val result = Array(1) { ByteArray(labelList.size) }
interpreter.run(byteBuffer, result)

val pq = PriorityQueue(
 MAX_RESULTS,
 Comparator<Classifier.Recognition> {...})

for (i in labelList.indices) {
 val confidence = labelProbArray[0][i]
 if (confidence >= THRESHOLD) {
 pq.add(Classifier.Recognition("" + i,
 if (labelList.size > i) labelList[i] else "Unknown", confidence))
 }
}

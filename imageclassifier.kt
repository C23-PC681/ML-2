// Preparing Input Classifier
Bitmap.createScaledBitmap(bitmap, 224, 224, false)
val byteBuffer = ByteBuffer.allocateDirect(4 *
 224 * 224 * 3)
byteBuffer.order(ByteOrder.nativeOrder())
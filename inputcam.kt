//Input Image
val image: Image = it.acquireLatestImage()
val planes = image.getPlanes()

yRowStride = planes[0].getRowStride()
uvRowStride = planes[1].getRowStride()
uvPixelStride = planes[1].getPixelStride()

// Converting ARGB
ImageUtils.convertYUV420ToARGB8888(
yuvBytes[0],yuvBytes[1],yuvBytes[2],previewSize.width,previewSize.height,
yRowStride,uvRowStride,uvPixelStride,rgbBytes)


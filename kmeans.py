#now go for the kmeans
# print('Using kmeans algorithm, it is faster!')
# km = MiniBatchKMeans(n_clusters = 6,random_state=0)
# km.fit(flatImg)
# labels = km.labels_

# # Displaying segmented image
# segmentedImg = np.reshape(labels, originShape[:2])
# print(len(segmentedImg))
# for x in range(0,len(segmentedImg)):
#   for y in range(0,len(segmentedImg[0])):
#     if segmentedImg[x,y] != 3:
#       segmentedImg[x,y] = 0
# segmentedImg = label2rgb(segmentedImg) * 255 # need this to work with cv2. imshow
# print(segmentedImg.shape)
# plt.imshow(segmentedImg.astype(np.uint8))
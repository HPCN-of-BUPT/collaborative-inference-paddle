import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# zip time
# gzip_ziptime = np.array([0.1,0.4,0.56,1.61,2.82,3.47,6.02])
# bz2_ziptime = np.array([0.32,1.25,1.76,4.93,8.99,10.55,19.46])
# lzma_ziptime = np.array([1.15,6.11,9.04,27.27,51.4,59.65,110.8])
# zipfile_ziptime = np.array([0.1,0.4,0.57,1.67,3.03,3.47,6.26])
# tarfile_ziptime = np.array([0.1,0.4,0.58,1.62,3.08,3.31,6.09])
# x = np.array([3,11.8,17,46.8,87.3,102,184])

# plt.plot(x,gzip_ziptime,label='gzip',marker='o',markersize=5)
# plt.plot(x,bz2_ziptime,label='bz2',marker='P',markersize=5)
# plt.plot(x,lzma_ziptime,label='lzma',marker='X',markersize=5)
# plt.plot(x,zipfile_ziptime,label='zipfile',marker='D',markersize=5)
# plt.plot(x,tarfile_ziptime,label='tarfile',marker='^',markersize=5)

# plt.xlabel('File Size(MB)')
# plt.ylabel('Zip Time(s)')

# zip ratio
# gzip_zipratio = np.array([0.93,0.93,0.93,0.93,0.93,0.93,0.93])
# bz2_zipratio = np.array([0.96,0.95,0.95,0.95,0.95,0.95,0.95])
# lzma_zipratio = np.array([0.95,0.92,0.95,0.92,0.92,0.92,0.92])
# zipfile_zipratio = np.array([0.93,0.93,0.93,0.93,0.93,0.93,0.93])
# tarfile_zipratio = np.array([0.93,0.93,0.93,0.93,0.93,0.93,0.93])
# x = np.array([3,11.8,17,46.8,87.3,102,184])

# plt.plot(x,gzip_zipratio,label='gzip',marker='o',markersize=5)
# plt.plot(x,bz2_zipratio,label='bz2',marker='P',markersize=5)
# plt.plot(x,lzma_zipratio,label='lzma',marker='X',markersize=5)
# plt.plot(x,zipfile_zipratio,label='zipfile',marker='D',markersize=5)
# plt.plot(x,tarfile_zipratio,label='tarfile',marker='^',markersize=5)

# plt.xlabel('File Size(MB)')
# plt.ylabel('Zip Ratio(after zip/before zip)')

# unzip time
gzip_unziptime = np.array([0.02,0.07,0.1,0.27,0.53,0.58,1.0])
bz2_unziptime = np.array([0.16,0.63,0.91,2.41,4.32,5.02,9.02])
lzma_unziptime = np.array([0.31,1.29,1.77,5.0,8.96,10.42,19])
zipfile_unziptime = np.array([0.02,0.06,0.09,0.23,0.48,0.53,0.97])
tarfile_unziptime = np.array([0.03,0.12,0.17,0.49,0.91,1.02,1.85])
x = np.array([3,11.8,17,46.8,87.3,102,184])

plt.plot(x,gzip_unziptime,label='gzip',marker='o',markersize=5)
plt.plot(x,bz2_unziptime,label='bz2',marker='P',markersize=5)
plt.plot(x,lzma_unziptime,label='lzma',marker='X',markersize=5)
plt.plot(x,zipfile_unziptime,label='zipfile',marker='D',markersize=5)
plt.plot(x,tarfile_unziptime,label='tarfile',marker='^',markersize=5)

plt.xlabel('File Size(MB)')
plt.ylabel('Unzip Time(s)')
plt.legend()
plt.show()
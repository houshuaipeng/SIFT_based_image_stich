// opencv_empty_proj.cpp : 定义控制台应用程序的入口点。  
//
   
#include <opencv2/opencv.hpp>  
#include <opencv2/features2d/features2d.hpp>  
#include<opencv2/nonfree/nonfree.hpp>  
#include<opencv2/legacy/legacy.hpp>  
#include<vector>  
using namespace std;  
using namespace cv;  
   
/* 交叉检查算法
 * 该算法进行两次匹配，第一次匹配可以使用前两种的匹配算法，第二次匹配时，使用的匹配算法的执行顺序与第一次匹配的顺序相反，
 * 将第二幅图像的每个关键点逐个与第一幅图像的全部关键点进行比较。
 * 只有两个方向上都匹配到了同一对特征点，才认为是一个有效的匹配对。
*/

void symmetryTest(std::vector<cv::DMatch>& matches1,
                       std::vector<cv::DMatch>& matches2,
                       std::vector<cv::DMatch>& symMatches)
{
   // 遍历图像1到图像2的匹配
   for (std::vector<cv::DMatch>::const_iterator matchIterator1= matches1.begin();
        matchIterator1!= matches1.end(); ++matchIterator1)
   {
       //  遍历图像2到图像1的匹配
       for (std::vector<cv::DMatch>::const_iterator matchIterator2= matches2.begin();
           matchIterator2!= matches2.end(); ++matchIterator2)
       {
           // 进行匹配测试
           if (matchIterator1->queryIdx == matchIterator2->trainIdx  &&
               matchIterator2->queryIdx == matchIterator1->trainIdx)
           {
               // 若是最好匹配，则加入
               symMatches.push_back(*matchIterator1);
               break;
           }
       }
   }
}

//优选匹配点
vector<DMatch> chooseGood(Mat descriptor,vector<DMatch> matches)
{
	double max_dist = 0; double min_dist = 100;
	 for( int i = 0; i < descriptor.rows; i++ )
		{ double dist = matches[i].distance;
		  if( dist < min_dist ) 
			  min_dist = dist;
		  if( dist > max_dist ) 
			  max_dist = dist;
	    }
	vector<DMatch> goodMatches;
	for(int i=0;i<descriptor.rows;i++)
	{
		if(matches[i].distance<3*min_dist)
			goodMatches.push_back(matches[i]);
	}
	return goodMatches;
}
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}
int main(int argc, char* argv[])  
{  
    const char* imagename = "C:/Users/14518/Desktop/毕业设计/用到的图片/testt.jpg";  
    
    //从文件中读入图像  
    Mat img = imread(imagename);  
    Mat img2=imread("C:/Users/14518/Desktop/毕业设计/用到的图片/test.jpg");   

    //显示图像  
	namedWindow("image before",0);
    imshow("image before", img);  
	namedWindow("image2 before",0);
    imshow("image2 before",img2);  
       
   
    //sift特征检测  
    SiftFeatureDetector  siftdtc;  
    vector<KeyPoint>kp1,kp2;  
   
    siftdtc.detect(img,kp1);  
    Mat outimg1;  
    drawKeypoints(img,kp1,outimg1);  
    //imshow("image1 keypoints",outimg1);  
   
    siftdtc.detect(img2,kp2);  
    Mat outimg2;  
    drawKeypoints(img2,kp2,outimg2);   
    //imshow("image2 keypoints",outimg2);  
   
   //在检测到的特征点上生成特征描述符
    SiftDescriptorExtractor extractor;  
    Mat descriptor1,descriptor2;  
    extractor.compute(img,kp1,descriptor1);  //第一个描述符
    extractor.compute(img2,kp2,descriptor2);  //第二个描述符
   
	FlannBasedMatcher matcher;
	//BFMatcher matcher;
	//vector<Mat> train_dest_collection(1,descriptor1);
	//matcher.add(train_dest_collection);
	//matcher.train();
    
	vector<DMatch> matches1,matches2;   //定义连接对象
    matcher.match(descriptor1,descriptor2,matches1);  //生成匹配对
	matcher.match(descriptor2,descriptor1,matches2);
   //匹配
	vector<DMatch> goodMatches1,goodMatches2,symMatches;
	goodMatches1=chooseGood(descriptor1,matches1);
	goodMatches2=chooseGood(descriptor2,matches2);

	Mat img_matches;
	symmetryTest(goodMatches1,goodMatches2,symMatches);
	drawMatches(img,kp1,img2,kp2,symMatches,img_matches);
	namedWindow("matches",CV_WINDOW_NORMAL);
    imshow("matches",img_matches);  

	/*图像配准*/
	//首先将点集转化为Point2f类型
	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i<symMatches.size(); i++)
	{
		imagePoints2.push_back(kp1[symMatches[i].queryIdx].pt);
		imagePoints1.push_back(kp2[symMatches[i].trainIdx].pt);
	}
	//开始实现配准
	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵     

	//计算配准图的四个顶点坐标
	CalcCorners(homo, img2);
	cout << "left_top:" << corners.left_top << endl;
	cout << "left_bottom:" << corners.left_bottom << endl;
	cout << "right_top:" << corners.right_top << endl;
	cout << "right_bottom:" << corners.right_bottom << endl;

	//图像配准  
	Mat imageTransform1, imageTransform2;
	warpPerspective(img2, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), img.rows));
	//warpPerspective(img2, imageTransform2, adjustMat*homo, Size(img.cols*1.3, img.rows*1.8));
	//imshow("直接经过透视矩阵变换", imageTransform1);
	imwrite("trans1.jpg", imageTransform1);

   
	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
	int dst_height = img.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	img.copyTo(dst(Rect(0, 0, img.cols, img.rows)));

	//imshow("b_dst", dst);


	OptimizeSeam(img, imageTransform1, dst);


	imshow("dst", dst);
	imwrite("dst.jpg", dst);

    //此函数等待按键，按键盘任意键就返回  
    waitKey();  
    return 0;  
}
//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <mutex>
#include <barrier>

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;

unsigned char blur(int x, int y, int channel, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset_y = -KERNEL_RADIUS; offset_y <= KERNEL_RADIUS; offset_y++)
	{
		for (int offset_x = -KERNEL_RADIUS; offset_x <= KERNEL_RADIUS; offset_x++)
		{
			int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
			int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
			int pixel = pixel_y * width + pixel_x;

			float weight = std::exp(-(offset_x * offset_x + offset_y * offset_y) / (2.f * sigma * sigma));

			ret += weight * input[4 * pixel + channel];
			sum_weight += weight;
		}
	}
	ret /= sum_weight;

	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

unsigned char blurAxis(int x, int y, int channel, int axis/*0: horizontal axis, 1: vertical axis*/, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
	{
		int offset_x = axis == 0 ? offset : 0;
		int offset_y = axis == 1 ? offset : 0;
		int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
		int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
		int pixel = pixel_y * width + pixel_x;

		float weight = std::exp(-(offset * offset) / (2.f * sigma * sigma));

		ret += weight * input[4 * pixel + channel];
		sum_weight += weight;
	}
	ret /= sum_weight;

	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

int gaussian_blur_serial(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of [width * height * number of channels]. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Perform Gaussian Blur to each pixel
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blur(x, y, channel, img_in, width, height);
			}
		}
	}

	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur - Serial: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("blurred_image_serial.jpg", width, height, 4, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_out;
	return time;
}

int gaussian_blur_separate_serial(const char* filename)
{
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load an image into an array of unsigned chars that is the size of width * height * number of channels. The channels are the Red, Green, Blue and Alpha channels of the image.
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	// Timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	// Horizontal Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
			}
		}
	}
	// Vertical Blur
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
			}
		}
	}
	// Timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Serial: Time %dms\n", time);

	// Write the blurred image into a JPG file
	stbi_write_jpg("blurred_separate.jpg", width, height, 4/*channels*/, img_out, 90 /*quality*/);

	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
	return time;
}


void parallel_func(int start_row, int end_row, int width, int height, int channels, unsigned char* img_in, unsigned char* img_out) {
	
	// Perform Gaussian Blur to each pixel within this thread's limits
	// Note that the start_row is actually included in this thread's area. However, the end_row is not and the counter stops one row before it.
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blur(x, y, channel, img_in, width, height);
			}
		}
	}
}

int gaussian_blur_parallel(const char* filename,const int thread_count /* Number of threads to run */)
{
	// Load an image into an array of unsigned chars that is the size of [width * height * number of channels]. The channels are the Red, Green, Blue and Alpha channels of the image.
	int width = 0; 
	int height = 0; 
	int channels = 4; 
	
	unsigned char* img_in = stbi_load(filename, &width, &height, &channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_out = new unsigned char[width * height * 4];


	// Start timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	int rows_remainder = height % thread_count;
	int rows_quotient = height / thread_count;
	int current_thread_rows = (rows_quotient) + 1;
	int offset = 0;
	int start_row = 0;
	int end_row = 0;
	
	std::vector<std::thread> threads;

	for (int i = 0; i < thread_count; i++) {

		if (i == rows_remainder) { current_thread_rows--; }
		


		start_row = (rows_quotient * i) + offset; //Calculate the start of this thread's area. Offset accounts for the extra rows appointed to previous threads. 
		end_row = start_row + current_thread_rows;

		
		threads.push_back(std::thread(parallel_func,start_row, end_row, width, height, channels, *&img_in, *&img_out));

		

		if (offset < rows_remainder) { offset++; }
	}
	
	for (int i = 0; i < thread_count; i++) {
		threads[i].join();
	}


	// End timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur - Parallel: Time %dms\n", time);
	

	// Write the blurred image into a JPG file
	stbi_write_jpg("blurred_image_parallel.jpg", width, height, 4, img_out, 90 /*quality*/);

	
	stbi_image_free(img_in);
	delete[] img_out;
	return time;
}

void time_tester_parallel() {
	
	int parallel_2_threads = 0 ;
	int parallel_4_threads = 0 ;
	int parallel_8_threads = 0 ;
	int serial_times = 0;
	
	const char* filename = "garden.jpg";
	
	for (int i = 0; i < 4; i++) {
		parallel_2_threads += gaussian_blur_parallel(filename, 2);
	}
	for (int i = 0; i < 4; i++) {
		parallel_4_threads += gaussian_blur_parallel(filename, 4);
	}
	for (int i = 0; i < 4; i++) {
		parallel_8_threads += gaussian_blur_parallel(filename, 8);
	}
	for (int i = 0; i < 4; i++) {
		serial_times += gaussian_blur_serial(filename);
	}

	double avg_2_threads = ((double)parallel_2_threads) / 4;
	double avg_4_threads = ((double)parallel_4_threads) / 4;
	double avg_8_threads = ((double)parallel_8_threads) / 4;
	double avg_serial_times = ((double)serial_times) / 4;

	std::printf("Average run time for 2 threads: %10f ms\n", avg_2_threads);
	std::printf("Average run time for 4 threads: %10f ms\n", avg_4_threads);
	std::printf("Average run time for 8 threads: %10f ms\n", avg_8_threads);
	std::printf("Average serial run time: %10f ms\n", avg_serial_times);
}

std::mutex max_values_mutex;
std::barrier bar{ 4 };

void separate_parallel_func( int start_row, int end_row, int width, int height, int channels, unsigned char* max_values, unsigned char* img_in, unsigned char* img_norm, unsigned char* img_horiz, unsigned char* img_out) {

	unsigned char local_max_values[4] = {0,0,0,0};

	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				
				if (img_in[4 * pixel + channel] > local_max_values[channel]) {
					local_max_values[channel] = img_in[4 * pixel + channel];
				}
			}
		}
	}

	max_values_mutex.lock();
	for (int channel = 0; channel < 4; channel++)
	{

		if (max_values[channel] < local_max_values[channel]) {
			max_values[channel] = local_max_values[channel];
		}
	}
	max_values_mutex.unlock();
	bar.arrive_and_wait();

	//Normalise image
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_norm[4 * pixel + channel] = 255 * img_in[4 * pixel + channel] / max_values[channel];
			}
		}
	}
	bar.arrive_and_wait();

	//Blur horizontal
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_horiz[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_norm, width, height);
			}
		}
	}
	bar.arrive_and_wait();

	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horiz, width, height);
			}
		}
	}
	bar.arrive_and_wait();

	std::cout << "Thread done" << std::endl;
}

void separate_parallel_func3(int start_row, int end_row, int width, int height, int channels, unsigned char* max_values, unsigned char* img_in, unsigned char* img_norm, unsigned char* img_horiz, unsigned char* img_out) {

	unsigned char local_max_values[4] = { 0,0,0,0 };

	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{

				if (img_in[4 * pixel + channel] > local_max_values[channel]) {
					local_max_values[channel] = img_in[4 * pixel + channel];
				}
			}
		}
	}

	max_values_mutex.lock();
	for (int channel = 0; channel < 4; channel++)
	{

		if (max_values[channel] < local_max_values[channel]) {
			max_values[channel] = local_max_values[channel];
		}
	}
	max_values_mutex.unlock();
	bar.arrive_and_wait();

	//Normalisation
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_norm[4 * pixel + channel] = 255 * img_in[4 * pixel + channel] / max_values[channel];
			}
		}
	}
	bar.arrive_and_wait();

	// Write the normalized image into a JPG file
	stbi_write_jpg("normalized_image_parallel.jpg", width, height, 4, img_norm, 90 /*quality*/);

	

	//Blur Horizontal
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_horiz[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_norm, width, height);
			}
		}
	}
	bar.arrive_and_wait();

	// Write the horizontaly blurred image into a JPG file
	stbi_write_jpg("image_blurred_horizontal.jpg", width, height, 4, img_horiz, 90 /*quality*/);

	

	//Blur vertical
	for (int y = start_row; y < end_row; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int pixel = y * width + x;
			for (int channel = 0; channel < 4; channel++)
			{
				img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horiz, width, height);
			}
		}
	}

	bar.arrive_and_wait();

	// Write the blurred image into a JPG file
	stbi_write_jpg("image_blurred_final.jpg", width, height, 4, img_out, 90 /*quality*/);

	std::cout << "Thread 3 done" << std::endl;
}

int gaussian_blur_separate_parallel(const char* filename) {
	
	// Load an image into an array of unsigned chars that is the size of [width * height * number of channels]. The channels are the Red, Green, Blue and Alpha channels of the image.
	int width = 0;
	int height = 0;
	int channels = 4;

	unsigned char* img_in = stbi_load(filename, &width, &height, &channels /*image file channels*/, 4 /*requested channels*/);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return 0;
	}

	unsigned char* img_norm = new unsigned char[width * height * 4];
	unsigned char* img_horiz = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];

	unsigned char* max_values = new unsigned char[4];
	for (int i = 0; i < 4; i++) {
		max_values[i] = 0;
	}

	// Start timer to measure performance
	auto start = std::chrono::high_resolution_clock::now();

	int rows_remainder = height % 4;
	int rows_quotient = height / 4;
	int current_thread_rows = (rows_quotient)+1;
	int offset = 0;
	int start_row = 0;
	int end_row = 0;

	std::vector<std::thread> threads;

	for (int i = 0; i < 4; i++) {

		if (i == rows_remainder) { current_thread_rows--; }


		start_row = (rows_quotient * i) + offset; //Calculate the start of this thread's area. Offset accounts for the extra rows appointed to previous threads. 
		end_row = start_row + current_thread_rows;

		if (i == 3) {
			threads.push_back(std::thread(separate_parallel_func3, start_row, end_row, width, height, channels, *&max_values, *&img_in, *&img_norm, *&img_horiz, *&img_out));
		}
		else {
			threads.push_back(std::thread(separate_parallel_func, start_row, end_row, width, height, channels, *&max_values, *&img_in, *&img_norm, *&img_horiz, *&img_out));
		}


		if (offset < rows_remainder) { offset++; }
	}

	for (int i = 0; i < 4; i++) {
		threads[i].join();
	}
	
	// End timer to measure performance
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur - Parallel: Time %dms\n", time);



	stbi_image_free(img_in);
	delete[] img_norm;
	delete[] img_horiz;
	delete[] img_out;
	delete[] max_values;
	return time;
}

void time_tester_separate_parallel() {

	int parallel_4_threads = 0;
	int serial_times = 0;

	const char* filename = "street_night.jpg";


	for (int i = 0; i < 4; i++) {
		parallel_4_threads += gaussian_blur_separate_parallel(filename);
	}

	for (int i = 0; i < 4; i++) {
		serial_times += gaussian_blur_separate_serial(filename);
	}


	double avg_4_threads = ((double)parallel_4_threads) / 4;
	double avg_serial_times = ((double)serial_times) / 4;

	std::printf("Average run time for 4 threads: %10f ms\n", avg_4_threads);
	std::printf("Average serial run time: %10f ms\n", avg_serial_times);
}

int main()
{
	const char* filename = "garden.jpg";
	//gaussian_blur_serial(filename);

	gaussian_blur_parallel(filename,4);

	//time_tester_parallel();
	
	const char* filename2 = "street_night.jpg";
	gaussian_blur_separate_serial(filename2);

	//gaussian_blur_separate_parallel(filename2);
	//time_tester_separate_parallel();

	return 0;
}
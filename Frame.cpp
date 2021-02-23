#include <SDL.h>
#include <iostream>
#include <chrono>
#include <array>
#include <immintrin.h>
#include <thread>
#include <vector>





class Frame{

public:
	int frame_width;
	int frame_height;
	
	double x_min;
	double y_min;
	
	double real_width;
	double real_height;
	
	double delta_x;
	double delta_y;
	
	int *counters;
	
	SDL_Renderer *renderer = NULL;
	SDL_Window *window = NULL;
	SDL_Texture *texture = NULL;
	
	uint32_t *pixels;

public:
	Frame(int width, int height){
		frame_width = width;
		frame_height = height;
		
		counters = new int[frame_width * frame_height];
		pixels = new uint32_t[frame_width * frame_height];
		
		y_min = -1.2f;
		// y_max = 1.0f
		x_min = y_min * (double(frame_width) / frame_height);
		
		real_width = -2.0 * x_min;
		real_height = -2.0 * y_min;
		
		x_min *= 1.3f;
		
		delta_x = real_width / frame_width;
		delta_y = real_height / frame_height;
		
		std::cout << "delta_x: " << delta_x << std::endl;
		
		SDL_Init(SDL_INIT_VIDEO);
		//SDL_CreateWindowAndRenderer(frame_width, frame_height, 0, &window, &renderer);
		
		window = SDL_CreateWindow("Mandelbot set", SDL_WINDOWPOS_CENTERED, 
								  SDL_WINDOWPOS_CENTERED, frame_width, frame_height, 0);
		
		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
		
		texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, 
									frame_width, frame_height);
		
		SDL_SetRenderDrawColor(renderer, 0,0,0,0);
		SDL_RenderClear(renderer);
		SDL_RenderPresent(renderer);
		
		
	}
	
	~Frame(){
		delete counters;
		delete pixels;
		SDL_DestroyTexture(texture);
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
	}
	
	void calculate_mandelbrot(int max_iterations){
		double x_step = delta_x;
		double y_step = delta_y;
		double x_pos = x_min;
		double y_pos = y_min;
		int y_offset = 0;
		
		double u, v, cr, ci;
		double temp;
		int n = 0;
		
		for(int y = 0; y < frame_height; y ++){
			
			x_pos = x_min;
			ci = y_pos;
			for(int x = 0; x < frame_width; x++){
			
				n = 0;
				u = 0.0;
				v = 0.0;
				cr = x_pos;

				while(n < max_iterations && u*u + v*v < 4.0){
					temp = u*u - v*v + cr;
					v = 2.0 * u * v + ci;
					u = temp;
					n ++;
				}
				
				x_pos += x_step;
				
				counters[x + y_offset] = n;
			}
			
			y_pos += y_step;
			y_offset += frame_width;
		
		}	
	}
	
	void calculate_mandelbrot_simd(int first_row, int last_row, int max_iterations){
		
		__m256d _u, _v, _u2, _v2, _cr, _ci, _2, _4;
		__m256d _x_pos, _x_offset, _x_step, _temp, _mask1;
		__m256i _1, _iterations, _n, _mask2, _c;
		double x_step = delta_x;
		double y_step = delta_y;
		double x_pos = x_min;
		double y_pos = y_min;
		
		//int y_offset = 0;
		int y_offset = first_row * frame_width;
		int rows = frame_height;
		int columns = frame_width;
		
		_1 = _mm256_set1_epi64x(1);
		_2 = _mm256_set1_pd(2.0);
		_4 = _mm256_set1_pd(4.0);
		_iterations = _mm256_set1_epi64x(max_iterations);
		
		_x_offset = _mm256_set_pd(0.0, x_step, 2.0 * x_step, 3.0 * x_step);
		_x_step = _mm256_set1_pd(4.0 * x_step);
		
		//for(int y = 0; y < rows; y++){
		for(int y = first_row; y < last_row; y++){
		
			// Initialize _x_pos = x_min
			_x_pos = _mm256_set1_pd(x_pos);
			_x_pos = _mm256_add_pd(_x_pos, _x_offset);
			
			_ci = _mm256_set1_pd(y_pos);
			
			for(int x = 0; x < columns; x += 4){
			
				_cr = _x_pos;
				_n = _mm256_setzero_si256();
				_u = _mm256_setzero_pd();
				_v = _mm256_setzero_pd();
				
				repeat:
				
					_u2 = _mm256_mul_pd(_u, _u);
					_v2 = _mm256_mul_pd(_v, _v);
					
					_temp = _mm256_sub_pd(_u2, _v2);
					_temp = _mm256_add_pd(_temp, _cr);
					_v = _mm256_mul_pd(_v, _u);
					_v = _mm256_mul_pd(_v, _2);
					_v = _mm256_add_pd(_v, _ci);
					
					_u = _temp;
					
					// Store the magnitude of z
					_temp = _mm256_add_pd(_u2, _v2);
					// u*u 0 v*v < 4.0
					_mask1 = _mm256_cmp_pd(_temp, _4, _CMP_LT_OQ);
					// If n < iterations
					_mask2 = _mm256_cmpgt_epi64(_iterations, _n);
					// n < iteration && u*u + v*v < 4.0
					_mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));
					
					_c = _mm256_and_si256(_1, _mask2); 
					// Increment counters
					_n = _mm256_add_epi64(_n, _c);
					
					if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0){
						goto repeat;
					}
					
				counters[x + y_offset] = int(_n[3]);
				counters[x + y_offset + 1] = int(_n[2]);
				counters[x + y_offset + 2] = int(_n[1]);
				counters[x + y_offset + 3] = int(_n[0]);
				
				_x_pos = _mm256_add_pd(_x_pos, _x_step);
			}
			
			y_pos += y_step;
			y_offset += columns;
		
		}

	}
	
	
	static void simd_mandelbrot(double x_min, 
						 double y_min, 
						 double delta_x, 
						 double delta_y, 
						 int frame_width, 
						 int frame_height, 
						 int first_row, 
						 int last_row, 
						 int *counters,
						 int max_iterations){
		
		__m256d _u, _v, _u2, _v2, _cr, _ci, _2, _4;
		__m256d _x_pos, _x_offset, _x_step, _temp, _mask1;
		__m256i _1, _iterations, _n, _mask2, _c;
		double x_step = delta_x;
		double y_step = delta_y;
		double x_pos = x_min;
		double y_pos = y_min + first_row * delta_y;
		
		//int y_offset = 0;
		int y_offset = first_row * frame_width;
		int rows = frame_height;
		int columns = frame_width;
		
		_1 = _mm256_set1_epi64x(1);
		_2 = _mm256_set1_pd(2.0);
		_4 = _mm256_set1_pd(4.0);
		_iterations = _mm256_set1_epi64x(max_iterations);
		
		_x_offset = _mm256_set_pd(0.0, x_step, 2.0 * x_step, 3.0 * x_step);
		_x_step = _mm256_set1_pd(4.0 * x_step);
		
		//for(int y = 0; y < rows; y++){
		for(int y = first_row; y < last_row; y++){
		
			// Initialize _x_pos = x_min
			_x_pos = _mm256_set1_pd(x_pos);
			_x_pos = _mm256_add_pd(_x_pos, _x_offset);
			
			_ci = _mm256_set1_pd(y_pos);
			
			for(int x = 0; x < columns; x += 4){
			
				_cr = _x_pos;
				_n = _mm256_setzero_si256();
				_u = _mm256_setzero_pd();
				_v = _mm256_setzero_pd();
				
				repeat:
				
					_u2 = _mm256_mul_pd(_u, _u);
					_v2 = _mm256_mul_pd(_v, _v);
					
					_temp = _mm256_sub_pd(_u2, _v2);
					_temp = _mm256_add_pd(_temp, _cr);
					_v = _mm256_mul_pd(_v, _u);
					_v = _mm256_mul_pd(_v, _2);
					_v = _mm256_add_pd(_v, _ci);
					
					_u = _temp;
					
					// Store the magnitude of z
					_temp = _mm256_add_pd(_u2, _v2);
					// u*u 0 v*v < 4.0
					_mask1 = _mm256_cmp_pd(_temp, _4, _CMP_LT_OQ);
					// If n < iterations
					_mask2 = _mm256_cmpgt_epi64(_iterations, _n);
					// n < iteration && u*u + v*v < 4.0
					_mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));
					
					_c = _mm256_and_si256(_1, _mask2); 
					// Increment counters
					_n = _mm256_add_epi64(_n, _c);
					
					if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0){
						goto repeat;
					}
					
				counters[x + y_offset    ] = int(_n[3]);
				counters[x + y_offset + 1] = int(_n[2]);
				counters[x + y_offset + 2] = int(_n[1]);
				counters[x + y_offset + 3] = int(_n[0]);
				
				_x_pos = _mm256_add_pd(_x_pos, _x_step);
			}
			
			y_pos += y_step;
			y_offset += columns;
		
		}

	}
	
	
	void calculate_mandelbrot_multi_threaded(int max_iterations, int num_threads){
		
		int rows_per_thread = frame_height / num_threads;
		
		std::vector<std::thread> thread_pool;
		
		for(int t = 0; t < num_threads; t++){
			thread_pool.push_back(std::thread(simd_mandelbrot, x_min, y_min, delta_x, delta_y,
											  frame_width, frame_height, t * rows_per_thread, 
											  (t + 1) * rows_per_thread, counters, max_iterations));
		}
		
		for(auto &thread : thread_pool){
			thread.join();
		}
	}
	
	void scale(double scale_factor){
		
		x_min += (1.0 - scale_factor) * real_width / 2.0;
		y_min += (1.0 - scale_factor) * real_height / 2.0;
		
		real_width *= scale_factor;
		real_height *= scale_factor;
		delta_x *= scale_factor;
		delta_y *= scale_factor;
		
		
	}
	
	void present(){
		
		for(int x = 0; x < frame_width; x++){
			for(int y = 0; y < frame_height; y++){
				SDL_SetRenderDrawColor(renderer, 0, 0, counters[x + y * frame_width],255);
				SDL_RenderDrawPoint(renderer, x, y);
			}
		}
		
		SDL_RenderPresent(renderer);
	}
	
	void fast_present(int max_iterations){
		for(int i = 0; i < frame_height; i++){
			for(int j = 0; j < frame_width; j++){
				int color_value = int(counters[i * frame_width + j] * 256.0 / max_iterations);
				pixels[i * frame_width + j] = color_value << 8;
			}
		}
		
		SDL_UpdateTexture(texture, NULL, pixels, frame_width * sizeof(uint32_t));
		SDL_RenderCopy(renderer, texture, NULL, NULL);
		SDL_RenderPresent(renderer);
	}

};



int main(int argc, char *argv[]){
	int width = 1920;
	int height = 1080;
	Frame frame(width, height);
	
	int max_iterations = 256;
	
	SDL_Event event;
	
	int counter = 0;
	while(event.type != SDL_QUIT){
		SDL_Delay(1);
		SDL_PollEvent(&event);
		if(event.type == SDL_QUIT){
			std::cout << "Quitting..." << std::endl;
		}
		
		double move_scale = 10.0;
		double zoom_scale = 0.1;
		const Uint8 *key_state = SDL_GetKeyboardState(NULL);
		if(key_state[SDL_SCANCODE_LEFT]){
			frame.x_min -= frame.delta_x * move_scale;
		}
		if(key_state[SDL_SCANCODE_RIGHT]){
			frame.x_min += frame.delta_x * move_scale;
		}
		if(key_state[SDL_SCANCODE_DOWN]){
			frame.y_min += frame.delta_y * move_scale;
		}
		if(key_state[SDL_SCANCODE_UP]){
			frame.y_min -= frame.delta_y * move_scale;
		}
		if(key_state[SDL_SCANCODE_LCTRL]){
			// zoom in
			frame.x_min += 0.5 * (zoom_scale) * frame.real_width;
			frame.y_min += 0.5 * (zoom_scale) * frame.real_height;
			frame.real_width *= 1.0 - zoom_scale;
			frame.real_height *= 1.0 - zoom_scale;
			frame.delta_x *= 1.0 - zoom_scale;
			frame.delta_y *= 1.0 - zoom_scale;
		}
		if(key_state[SDL_SCANCODE_LALT]){
			// zoom out
			frame.x_min -= 0.5 * (zoom_scale) * frame.real_width;
			frame.y_min -= 0.5 * (zoom_scale) * frame.real_height;
			frame.real_width *= 1.0 + zoom_scale;
			frame.real_height *= 1.0 + zoom_scale;
			frame.delta_x *= 1.0 + zoom_scale;
			frame.delta_y *= 1.0 + zoom_scale;
		}
		if(key_state[SDL_SCANCODE_1]){
			if(max_iterations > 16){
				max_iterations -= 16;
				std::cout << max_iterations << std::endl;
			}
		}
		if(key_state[SDL_SCANCODE_2]){
			max_iterations += 16;
			std::cout << max_iterations << std::endl;
		}
		
		////////////////////////////////////////////////////////////////////////////////////////////
		auto start_time = std::chrono::high_resolution_clock::now();
		
		//frame.calculate_mandelbrot(256);
		//frame.calculate_mandelbrot_simd(0, height, max_iterations);
		frame.calculate_mandelbrot_multi_threaded(max_iterations, 10);
		//frame.present();
		
		auto end_time = std::chrono::high_resolution_clock::now();
		auto math_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		//std::cout << "Frame time" << duration.count() << "ms" << std::endl;
		
		auto start = std::chrono::high_resolution_clock::now();
		
		//frame.present();
		frame.fast_present(max_iterations);
		
		auto end = std::chrono::high_resolution_clock::now();
		auto draw_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Frame time " << math_duration.count() << "ms " << draw_duration.count() << "ms" << std::endl;
		////////////////////////////////////////////////////////////////////////////////////////////
		//frame.scale(0.99);
		
		counter += 1;
		if(counter == 5000){
			//break;
		}
	}
}














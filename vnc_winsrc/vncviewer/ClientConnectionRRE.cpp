
#include <pthread.h>
#include <semaphore.h>
#include <list>

#include "stdhdrs.h"
#include "vncviewer.h"
#include "ClientConnection.h"

#include "CNNEncoder.h"
#include "huffman.h"


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ALIGN(x,y) (CEIL_DIV(x,y)*y)


typedef struct {
	CARD32 origin_width;
	CARD32 origin_height;
	CARD32 padded_width;
	CARD32 padded_height;
	CARD32 origin_x;
	CARD32 origin_y;
	CARD32 data_size;
}CNNEncBlock;

typedef struct {
	CARD32 block_num;
	CARD32 data_size;
	CARD32 decompressed_size;
	CARD32 checksum;
}rfbCNNHeader;

typedef struct {
	CNNEncBlock* block_info;
	char* origin_data;
	vector<char>* pixel_output;
	rfbPixelFormat* pixel_formate;
	sem_t* sem_finished;
} CNNDecTaskInfo;

#define TASK_THREAD_NUM 6

namespace cnn_static {
	pthread_mutex_t mutex_queue;
	std::list<CNNDecTaskInfo> task_queue;
	sem_t sem_queue;
	bool multi_thread_inited = false;

	void* cnn_task_thread_function(void* param);

	pthread_t task_threads[TASK_THREAD_NUM];

	void inline init_multi_thread() {
		if (!multi_thread_inited) {
			pthread_mutex_init(&mutex_queue, nullptr);
			sem_init(&sem_queue, 0, 0);
			for (int i = 0; i < TASK_THREAD_NUM; i++) {
				pthread_create(&task_threads[i], nullptr, cnn_task_thread_function, nullptr);
			}
			multi_thread_inited = true;
		}
	}

	void* cnn_task_thread_function(void* param) {
		OrtEnvInstance* p_ort = new OrtEnvInstance(false);
		CNNDecoder* p_decoder = new CNNDecoder(p_ort, "decoder_256.onnx", 480, 480, 192, nullptr);
		vector<char> model_input;
		vector<float> model_output;
		while (1) {
			sem_wait(&sem_queue);
			pthread_mutex_lock(&mutex_queue);
			CNNDecTaskInfo task = task_queue.front();
			task_queue.pop_front();
			pthread_mutex_unlock(&mutex_queue);

			const UINT X = task.block_info->origin_x;
			const UINT Y = task.block_info->origin_y;
			const UINT wid = task.block_info->origin_width;
			const UINT hei = task.block_info->origin_height;
			UINT W = task.block_info->padded_width;
			UINT H = task.block_info->padded_height;

			if (task.block_info->data_size == 4 || task.block_info->data_size == 2 || task.block_info->data_size == 1) {
				assert(task.block_info->data_size == (task.pixel_formate->bitsPerPixel / 8));

				if (task.pixel_formate->bitsPerPixel == 32) {
					CARD32 singlePixel = 0;
					memcpy(&singlePixel, task.origin_data, task.block_info->data_size);
					task.pixel_output->assign(wid * hei * sizeof(CARD32), 0);
					CARD32* p_buf = (CARD32*)(task.pixel_output->data());
					int offset = 0, stride = W * H;
					for (int h = 0; h < hei; h++) {
						for (int w = 0; w < wid; w++) {
							p_buf[offset++] = singlePixel;
						}
					}
				}
			}
			else {
				model_output.clear();
				if (W == 0 || H == 0) {
					W = wid;
					H = hei;
					model_output.assign(wid * hei * sizeof(float) * 3, 0);
					assert(task.block_info->data_size == model_output.size());
					memcpy(model_output.data(), task.origin_data, task.block_info->data_size);
				}
				else {
					model_input.assign(task.block_info->data_size, 0);
					memcpy(model_input.data(), task.origin_data, task.block_info->data_size);

					if (p_decoder->decode_chw(&model_input, &model_output, W, H) <= 0) {
						printf("Decoder error \n");
						exit(0);
					}
				};

				const UINT rm = task.pixel_formate->redMax;
				const UINT gm = task.pixel_formate->greenMax;
				const UINT bm = task.pixel_formate->blueMax;
				const UINT rs = task.pixel_formate->redShift;
				const UINT gs = task.pixel_formate->greenShift;
				const UINT bs = task.pixel_formate->blueShift;

				if (task.pixel_formate->bitsPerPixel == 32) {
					task.pixel_output->assign(wid * hei * sizeof(CARD32), 0);
					CARD32* p_buf = (CARD32*)(task.pixel_output->data());
					int offset = 0, stride = W * H;
					for (int h = 0; h < hei; h++) {
						for (int w = 0; w < wid; w++) {
							int in_offset = w + h * W;
							UINT red = (UINT)(model_output[in_offset] * rm) & rm;
							UINT green = (UINT)(model_output[in_offset + stride] * gm) & gm;
							UINT blue = (UINT)(model_output[in_offset + stride * 2] * bm) & bm;
							p_buf[offset++] = ((red << rs) | (green << gs) | (blue << bs));
						}
					}
				}
			}

			sem_post(task.sem_finished);
		}

		return nullptr;
	}
}



#define sz_rfcCNNHeader (sizeof(rfbCNNHeader))


void ClientConnection::ReadRRERect(rfbFramebufferUpdateRectHeader* pfburh)
{
	assert(m_myFormat.bitsPerPixel == 32);

	cnn_static::init_multi_thread();

	// An RRE rect is always followed by a background color
	// For speed's sake we read them together into a buffer.
	char tmpbuf[sz_rfcCNNHeader];			// biggest pixel is 4 bytes long
	rfbCNNHeader* prreh = (rfbCNNHeader*)tmpbuf;
	ReadExact(tmpbuf, sz_rfcCNNHeader);

	vector<char> compressed;
	compressed.assign(prreh->data_size, 0);
	ReadExact(compressed.data(), prreh->data_size);
	
	CARD32 cs = 0;
	for (BYTE b : compressed) cs = (cs + b) & 0xff;
	assert(cs == prreh->checksum);

	vector<char> decompressed;
	decompressed.reserve(prreh->decompressed_size);
	huffman_decompress(&compressed, &decompressed);

	SETUP_COLOR_SHORTCUTS;

	size_t data_offset = 0;
	vector<CNNEncBlock*> block_infos;
	vector<vector<char>> output_pixels;
	block_infos.assign(prreh->block_num, nullptr);
	output_pixels.assign(prreh->block_num, vector<char>());
	sem_t sem;
	sem_init(&sem, 0, 0);
	for (int i = 0; i < prreh->block_num; i++) {
		CNNDecTaskInfo task;
		CNNEncBlock* pblkh = (CNNEncBlock*)(decompressed.data() + data_offset);
		block_infos[i] = pblkh;
		task.block_info = pblkh;

		data_offset += sizeof(CNNEncBlock);

		task.origin_data = decompressed.data() + data_offset;
		data_offset += pblkh->data_size;

		task.pixel_formate = &m_myFormat;
		task.pixel_output = &output_pixels[i];
		task.sem_finished = &sem;

		pthread_mutex_lock(&cnn_static::mutex_queue);
		cnn_static::task_queue.push_back(task);
		pthread_mutex_unlock(&cnn_static::mutex_queue);
		sem_post(&cnn_static::sem_queue);
	}

	for (int i = 0; i < prreh->block_num; i++) {
		sem_wait(&sem);
	}
	sem_destroy(&sem);


	omni_mutex_lock l(m_bitmapdcMutex);
	ObjectSelector b(m_hBitmapDC, m_hBitmap);
	PaletteSelector p(m_hBitmapDC, m_hPalette);

	HDC srcdc = CreateCompatibleDC(m_hBitmapDC);
	for (int i = 0; i < prreh->block_num; i++) {
		if (m_myFormat.bitsPerPixel == 32) {
			HBITMAP bitmap = CreateBitmap(
				block_infos[i]->origin_width,
				block_infos[i]->origin_height,
				1, 32,
				output_pixels[i].data()
			);
			SelectObject(srcdc, bitmap);
			BitBlt(
				m_hBitmapDC,
				block_infos[i]->origin_x,
				block_infos[i]->origin_y,
				block_infos[i]->origin_width,
				block_infos[i]->origin_height,
				srcdc,
				0, 0,
				SRCCOPY
			);
		}
	}
	DeleteDC(srcdc);
}

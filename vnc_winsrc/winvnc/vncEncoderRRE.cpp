
#include <pthread.h>
#include <semaphore.h>

#include "vncEncodeRRE.h"
#include "CNNEncoder.h"
#include "huffman.h"


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ALIGN(x,y) (CEIL_DIV(x,y)*y)


vncEncodeRRE::vncEncodeRRE()
{
	m_buffer = NULL;
	m_bufflen = 0;
}

vncEncodeRRE::~vncEncodeRRE()
{
	if (m_buffer != NULL)
	{
		delete[] m_buffer;
		m_buffer = NULL;
	}
}

void
vncEncodeRRE::Init()
{
	vncEncoder::Init();
}

UINT
vncEncodeRRE::RequiredBuffSize(UINT width, UINT height)
{
	return vncEncoder::RequiredBuffSize(width, height);
}

UINT
vncEncodeRRE::NumCodedRects(RECT& rect)
{
	return 1;
}

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

#define sz_rfcCNNHeader (sizeof(rfbCNNHeader))

#define RectSizeValidate(r) ((r.right - r.left) <= 480 && (r.bottom - r.top) <= 480 && (r.right - r.left) * (r.bottom - r.top) <= 320 * 320)
#define AreaOfRect(r) ((r.right - r.left) * (r.bottom - r.top))

typedef struct {
	UINT            operation = 0;
	CNNEncBlock*     block_info;
	UINT            bytes_per_row;
	UINT            valid_wid;
	UINT            valid_hei;
	rfbPixelFormat*  pixel_formate;
	BYTE*           source;
	vector<char>*    model_output;
	sem_t*          sem_finished;
} CNNEncTaskInfo;

#define TASK_THREAD_NUM 6

namespace cnn_static {
	pthread_mutex_t mutex_queue;
	std::list<CNNEncTaskInfo> task_queue;
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

	template<class DataType, int BPP>
	void handel_cnn_encode_task(CNNEncTaskInfo& task, CNNEncoder *p_encoder) {
		const UINT X = task.block_info->origin_x;
		const UINT Y = task.block_info->origin_y;
		const UINT wid = task.block_info->origin_width;
		const UINT hei = task.block_info->origin_height;
		UINT W = task.block_info->padded_width;
		UINT H = task.block_info->padded_height;

		DataType* p_pix = (DataType*)(task.source + (Y * task.bytes_per_row + X * (BPP / 8)));
		DataType pix = *p_pix;
		bool solid = true;
		for (int y = 0; y < hei; y++) {
			for (int x = 0; x < wid; x++) {
				if (pix != p_pix[x]) {
					solid = false;
					break;
				}
			}
			p_pix += task.bytes_per_row / sizeof(DataType);
		}
		//pix = 0;//for debug
		if (solid) {
			task.model_output->assign(sizeof(DataType), 0);
			memcpy(task.model_output->data(), &pix, sizeof(DataType));
		}
		else {
			vector<float> model_input;
			bool raw = false;
			if (W == 0 || H == 0) {
				raw = true;
				W = wid;
				H = hei;
			}
			model_input.assign(W * H * 3, 0);
			int offset = 0;
			int ori_offset = 0;
			for (int h = 0; h < H; h++) {
				if (h < task.valid_hei) {
					offset = h * W;
					ori_offset = ((h + Y) * task.bytes_per_row) + X * (BPP / 8);
					DataType pixel = 0;
					for (int w = 0; w < W; w++) {
						if (w < task.valid_wid) {
							pixel = *((DataType*)(task.source + ori_offset));
							ori_offset += (BPP / 8);
						}
						model_input[offset] = (float)((pixel >> task.pixel_formate->redShift) & task.pixel_formate->redMax) / task.pixel_formate->redMax;
						model_input[offset + W * H] = (float)((pixel >> task.pixel_formate->greenShift) & task.pixel_formate->greenMax) / task.pixel_formate->greenMax;
						model_input[offset + W * H * 2] = (float)((pixel >> task.pixel_formate->blueShift) & task.pixel_formate->blueMax) / task.pixel_formate->blueMax;
						offset++;
					}
				}
				else {
					memcpy(model_input.data() + (h * W), model_input.data() + offset, sizeof(float) * W);
					memcpy(model_input.data() + (h * W) + (W * H), model_input.data() + offset + (W * H), sizeof(float) * W);
					memcpy(model_input.data() + (h * W) + (W * H * 2), model_input.data() + offset + (W * H * 2), sizeof(float) * W);
				}
			}

			if (raw) {
				task.model_output->assign(W * H * sizeof(float) * 3, 0);
				memcpy(task.model_output->data(), model_input.data(), task.model_output->size());
			}
			else if (p_encoder->encode_chw(&model_input, task.model_output, W, H) <= 0) {
				printf("Encode Error\n");
				exit(-1);
			}
		}
		task.block_info->data_size = task.model_output->size();
	}

	void* cnn_task_thread_function(void* param) {
		OrtEnvInstance* p_ort = new OrtEnvInstance(false);
		CNNEncoder* p_encoder = new CNNEncoder(p_ort, "encoder_256.onnx", 480, 480, 192, nullptr);
		while (1) {
			sem_wait(&sem_queue);
			pthread_mutex_lock(&mutex_queue);
			CNNEncTaskInfo task = task_queue.front();
			task_queue.pop_front();
			pthread_mutex_unlock(&mutex_queue);

			if (task.operation == 1) {
				memcpy(task.source, task.block_info, sizeof(CNNEncBlock));
				memcpy(task.source + sizeof(CNNEncBlock), task.model_output->data(), task.model_output->size());
				sem_post(task.sem_finished);
				continue;
			}

			switch (task.pixel_formate->bitsPerPixel) {
			case 32:
			case 24:
				handel_cnn_encode_task<CARD32, 32>(task, p_encoder);
				break;
			case 16:
				handel_cnn_encode_task<CARD16, 16>(task, p_encoder);
				break;
			case 8:
				handel_cnn_encode_task<CARD8, 8>(task, p_encoder);
				break;
			}

			sem_post(task.sem_finished);
		}

		return nullptr;
	}
}


inline UINT
vncEncodeRRE::EncodeRect(BYTE* source, BYTE* dest, const RECT& rect, int offx, int offy)
{
	//if (!m_localformat.bitsPerPixel == 32) {
	//	printf("Not support bpp %d\n", m_localformat.bitsPerPixel);
	//	exit(0);
	//}

	if (rect.left != 0x123456) {
		printf("Not support rectangle formate\n");
		exit(0);
	}

	std::list<RECT>** pp_list = (std::list<RECT>**)(&rect.top);
	std::list<RECT>* p_list = *pp_list;

	RECT main_rect = p_list->back();
	p_list->pop_back();
	if (main_rect.left || main_rect.top) {
		for (auto& r : *p_list) {
			r.left = r.left - main_rect.left;
			r.right = r.right - main_rect.left;
			r.top = r.top - main_rect.top;
			r.bottom = r.bottom - main_rect.top;
		}
	}

	UINT min_x = 0xffffff, min_y = 0xffffff, max_x = 0, max_y = 0;
	for (auto& r : *p_list) {
		max_x = max(max_x, r.right);
		max_y = max(max_y, r.bottom);
		min_x = min(min_x, r.left);
		min_y = min(min_y, r.top);
	}

	cnn_static::init_multi_thread();

	// Create the rectangle header
	rfbFramebufferUpdateRectHeader* surh = (rfbFramebufferUpdateRectHeader*)dest;
	surh->r.x = (CARD16)min_x;
	surh->r.y = (CARD16)min_y;
	surh->r.w = (CARD16)(max_x - min_x);
	surh->r.h = (CARD16)(max_y - min_y);
	surh->r.x = Swap16IfLE(surh->r.x - offx);
	surh->r.y = Swap16IfLE(surh->r.y - offy);
	surh->r.w = Swap16IfLE(surh->r.w);
	surh->r.h = Swap16IfLE(surh->r.h);
	surh->encoding = Swap32IfLE(rfbEncodingRRE);

	std::vector<RECT> rects_finished;
	rects_finished.reserve(20);
	for (auto& rec : *p_list) {
		if (RectSizeValidate(rec) || AreaOfRect(rec) <= 2048) {
			rects_finished.push_back(rec);
		}
		else {
			int x_num = CEIL_DIV(rec.right - rec.left, 320);
			int y_num = CEIL_DIV(rec.bottom - rec.top, 320);
			int x_step = (rec.right - rec.left) / x_num;
			int y_step = (rec.bottom - rec.top) / y_num;
			RECT r = rec;
			rects_finished.reserve(x_num * y_num);
			for (int y = 0; y < y_num; y++) {
				r.top = rec.top + y * y_step;
				r.bottom = (y == y_num - 1) ? rec.bottom : (r.top + y_step);
				for (int x = 0; x < x_num; x++) {
					r.left = rec.left + x * x_step;
					r.right = (x == x_num - 1) ? rec.right : (r.left + x_step);
					rects_finished.push_back(r);
				}
			}
		}
	}

	delete p_list;

	rfbCNNHeader* cnnheader = (rfbCNNHeader*)(dest + sz_rfbFramebufferUpdateRectHeader);
	cnnheader->block_num = rects_finished.size();
	cnnheader->decompressed_size = 0;

	vector<vector<char>> model_outputs;
	vector<CNNEncBlock> block_infos;
	vector<CNNEncTaskInfo> task_infos;
	block_infos.assign(rects_finished.size(), CNNEncBlock());
	model_outputs.assign(rects_finished.size(), vector<char>());
	task_infos.assign(rects_finished.size(), CNNEncTaskInfo());

	sem_t sem;
	sem_init(&sem, 0, 0);

	for (int i = 0; i < rects_finished.size(); i++) {
		block_infos[i].origin_x = rects_finished[i].left;
		block_infos[i].origin_y = rects_finished[i].top;
		block_infos[i].origin_width = rects_finished[i].right - rects_finished[i].left;
		block_infos[i].origin_height = rects_finished[i].bottom - rects_finished[i].top;
		if (AreaOfRect(rects_finished[i]) <= 100) {
			block_infos[i].padded_width = block_infos[i].padded_height = 0;
		}
		else {
			block_infos[i].padded_width = ALIGN(block_infos[i].origin_width, 16);
			block_infos[i].padded_height = ALIGN(block_infos[i].origin_height, 16);
		}

		CNNEncTaskInfo &task = task_infos[i];
		task.block_info = &block_infos[i];
		task.bytes_per_row = m_bytesPerRow;
		task.pixel_formate = &m_localformat;
		task.model_output = &model_outputs[i];
		task.sem_finished = &sem;
		task.source = source;
		task.valid_wid = max_x - rects_finished[i].left;
		task.valid_hei = max_y - rects_finished[i].top;

		pthread_mutex_lock(&cnn_static::mutex_queue);
		cnn_static::task_queue.push_back(task);
		pthread_mutex_unlock(&cnn_static::mutex_queue);
		sem_post(&cnn_static::sem_queue);
	}

	for (int i = 0; i < rects_finished.size(); i++) {
		sem_wait(&sem);
	}

	cnnheader->decompressed_size = 0;
	for (auto& vec : model_outputs) {
		cnnheader->decompressed_size += sizeof(CNNEncBlock);
		cnnheader->decompressed_size += vec.size();
	}

	vector<char> decompressed;
	decompressed.assign(cnnheader->decompressed_size, 0);

	size_t data_offset = 0;
	for (int i = 0; i < rects_finished.size(); i++) {
		task_infos[i].operation = 1;
		task_infos[i].source = (BYTE*)(decompressed.data()) + data_offset;
		task_infos[i].model_output = &(model_outputs[i]);
		task_infos[i].block_info = &(block_infos[i]);
		pthread_mutex_lock(&cnn_static::mutex_queue);
		cnn_static::task_queue.push_back(task_infos[i]);
		pthread_mutex_unlock(&cnn_static::mutex_queue);
		sem_post(&cnn_static::sem_queue);
		data_offset += sizeof(CNNEncBlock);
		data_offset += model_outputs[i].size();
	}

	for (int i = 0; i < rects_finished.size(); i++) {
		sem_wait(&sem);
	}
	sem_destroy(&sem);

	vector<char> compressed;
	huffman_compress(&decompressed, &compressed);

	BYTE* data_address = dest + sz_rfbFramebufferUpdateRectHeader + sz_rfcCNNHeader;
	memcpy(data_address, compressed.data(), compressed.size());

	cnnheader->data_size = compressed.size();

	CARD32 cs = 0;
	for (BYTE b : compressed) cs = (cs + b) & 0xff;
	cnnheader->checksum = cs;

	// Update statistics for this rectangle.
	rectangleOverhead += sz_rfbFramebufferUpdateRectHeader;
	dataSize += decompressed.size();
	encodedSize += sz_rfcCNNHeader + compressed.size();
	transmittedSize += sz_rfbFramebufferUpdateRectHeader + sz_rfcCNNHeader + compressed.size();

	// Return the amount of data sent	
	return sz_rfbFramebufferUpdateRectHeader + sz_rfcCNNHeader + compressed.size();
}



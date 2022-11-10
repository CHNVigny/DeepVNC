#include "CNNEncoder.h"
#include "huffman.h"

int CNNEncoder::encode_chw(std::vector<float> *data, std::vector<char> *output, int width, int height)
{
    int channel = 3;
    const char *inputName = p_session->GetInputName(0, allocator); // input
    const char *outputName = p_session->GetOutputName(0, allocator);

    std::vector<int64_t> inputDims = {1, channel, height, width};
    std::vector<int64_t> outputDims = {1, out_channel, height / 16, width / 16};
    size_t input_size = width * height * channel;
    size_t output_size = (width / 16) * (height / 16) * out_channel;

    Ort::Value input_data = Ort::Value::CreateTensor<float>(
        *p_memoryInfo, data->data(), input_size,
        inputDims.data(), inputDims.size());
    Ort::Value output_data = Ort::Value::CreateTensor<float>(
        *p_memoryInfo, out_buffer, output_size,
        outputDims.data(), outputDims.size());

    time_t start = clock();

    p_session->Run(
        Ort::RunOptions{nullptr},
        &inputName, &input_data, 1,
        &outputName, &output_data, 1);

    float *tmp = output_data.GetTensorMutableData<float>();

    time_t mid1 = clock();
    // printf("Model output size: %.3fKB\n", output_size * sizeof(float) / 1000.);
    // printf("Time of enc: %.3fs\n", double(mid1 - start) / CLOCKS_PER_SEC);
    if(logger) {
        logger->cnn_enc_time_s += double(mid1 - start) / CLOCKS_PER_SEC;
    }

    std::vector<char> byte_buf;
    byte_buf.assign(output_size, 0);
    for (size_t i = 0; i < output_size; i++)
    {
        int val = (int)(out_buffer[i]);
        if (val < -128)
            val = -128;
        else if (val > 127)
            val = 127;
        byte_buf[i] = val;
    }

    huffman_compress(&byte_buf, output);

    time_t mid2 = clock();
    // printf("Time of huffman-enc: %.3fs\n", double(mid2 - mid1) / CLOCKS_PER_SEC);
    if(logger) {
        logger->huf_enc_time_s += double(mid2 - mid1) / CLOCKS_PER_SEC;
        logger->enc_count ++;
        if(logger->on_finish_enc) logger->on_finish_enc(logger);
    }

    return output->size();
}

int CNNDecoder::decode_chw(std::vector<char> *data, std::vector<float> *output, int width, int height)
{
    int channel = 3;

    std::vector<int64_t> inputDims = {1, out_channel, height / 16, width / 16};
    std::vector<int64_t> outputDims = {1, channel, height, width};
    size_t input_size = (width / 16) * (height / 16) * out_channel;
    size_t output_size = width * height * channel;

    time_t start = clock();

    std::vector<char> byte_buf;
    huffman_decompress(data, &byte_buf);
    int i = 0;
    for (char c : byte_buf)
    {
        int val = static_cast<int>(c);
        in_buffer[i++] = val;
    }

    time_t mid1 = clock();
    // printf("Time of huffman-dec: %.3fs\n", double(mid1 - start) / CLOCKS_PER_SEC);
    if(logger) {
        logger->huf_dec_time_s += double(mid1 - start) / CLOCKS_PER_SEC;
    }

    const char *inputName = p_session->GetInputName(0, allocator); // input
    const char *outputName = p_session->GetOutputName(0, allocator);

    output->assign(output_size, 0);

    // printf("Dec model input size: %.3fKB\n", input_size * sizeof(float) / 1000.);
    Ort::Value input_data = Ort::Value::CreateTensor<float>(
        *p_memoryInfo, in_buffer, input_size,
        inputDims.data(), inputDims.size());
    Ort::Value output_data = Ort::Value::CreateTensor<float>(
        *p_memoryInfo, output->data(), output_size,
        outputDims.data(), outputDims.size());

    p_session->Run(
        Ort::RunOptions{nullptr},
        &inputName, &input_data, 1,
        &outputName, &output_data, 1);

    time_t mid2 = clock();
    // printf("Time of dec: %.3fs\n", double(mid2 - mid1) / CLOCKS_PER_SEC);
    if(logger) {
        logger->cnn_dec_time_s += double(mid2 - mid1) / CLOCKS_PER_SEC;
        logger->dec_count ++;
        if(logger->on_finish_dec) logger->on_finish_dec(logger);
    }

    float *tmp = output_data.GetTensorMutableData<float>();

    return output_size;
}

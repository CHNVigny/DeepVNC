
#pragma once

#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>

using std::vector;

class CNNEncLogger {
public:

    typedef void (*CNNEncLoggerCallback)(CNNEncLogger *);

    CNNEncLogger(){};
    ~CNNEncLogger(){};
    double cnn_enc_time_s = 0.;
    double huf_enc_time_s = 0.;
    double cnn_dec_time_s = 0.;
    double huf_dec_time_s = 0.;
    long enc_count = 0;
    long dec_count = 0;
    CNNEncLoggerCallback on_finish_enc = nullptr;
    CNNEncLoggerCallback on_finish_dec = nullptr;
};

class OrtEnvInstance {
public:
    OrtEnvInstance(bool useOPENVINO = false) : env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "encoder-decoder") {
        if (useOPENVINO)
        {
            std::cout << "Inference Execution Provider: OPENVINO" << std::endl;
        }
        else
        {
            std::cout << "Inference Execution Provider: CPU" << std::endl;
        }
        if (useOPENVINO) {
            OrtOpenVINOProviderOptions options;
            options.device_type = "CPU_FP32"; //Other options are: GPU_FP32, GPU_FP16, MYRIAD_FP16
            std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
            sessionOptions.AppendExecutionProvider_OpenVINO(options);
        }
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetIntraOpNumThreads(1);
        p_memoryInfo = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault)
        );
    }
    ~OrtEnvInstance() {
    }
    inline Ort::MemoryInfo* get_memory_info() {
        return p_memoryInfo;
    }
    inline Ort::Session* create_session(const char* model_path) {
        std::cout << "create model session : " << model_path << std::endl;
        std::string str_arg2(model_path);
        std::wstring wide_string_arg2 = std::wstring(str_arg2.begin(), str_arg2.end());
        std::basic_string<ORTCHAR_T> modelFilepath = std::basic_string<ORTCHAR_T>(wide_string_arg2);
        return new Ort::Session(env, modelFilepath.c_str(), sessionOptions);
    }
protected:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::MemoryInfo *p_memoryInfo;
};


class CNNEncoder {
public:
    CNNEncoder(OrtEnvInstance *env, const char *model_path, int max_width, int max_height, int out_channel, CNNEncLogger *log = nullptr)
    {
        this->logger = log;
        this->out_channel = out_channel;
        p_session = env->create_session(model_path);
        p_memoryInfo = env->get_memory_info();
        out_buffer = new float[(max_width / 16) * (max_height / 16) * out_channel];
    }

    ~CNNEncoder() {
        delete p_session;
        delete[] out_buffer;
    }

    int encode_chw(std::vector<float> *data, std::vector<char> *output, int width, int height);
protected:
    Ort::Session* p_session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo* p_memoryInfo;
    int out_channel;
    float* out_buffer;
    CNNEncLogger *logger;
};


class CNNDecoder {
public:
    CNNDecoder(OrtEnvInstance* env, const char* model_path, int max_width, int max_height, int out_channel, CNNEncLogger *log = nullptr)
    {
        this->logger = log;
        this->out_channel = out_channel;
        p_session = env->create_session(model_path);
        p_memoryInfo = env->get_memory_info();
        in_buffer = new float[(max_width / 16) * (max_height / 16) * out_channel];
    }

    ~CNNDecoder() {
        delete p_session;
        delete[] in_buffer;
    }

    int decode_chw(std::vector<char>* data, std::vector<float>* output, int width, int height);
protected:
    Ort::Session* p_session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo* p_memoryInfo;
    int out_channel;
    float* in_buffer;
    CNNEncLogger *logger;
};



#ifndef AGV_HUFFMAN_COMPRESS_H
#define AGV_HUFFMAN_COMPRESS_H

#include <vector>

int huffman_compress(std::vector<char> *input, std::vector<char> *output);

int huffman_decompress(std::vector<char> *input, std::vector<char> *output);

#endif


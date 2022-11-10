
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "BitIoStream.hpp"
#include "CanonicalCode.hpp"
#include "FrequencyTable.hpp"
#include "HuffmanCoder.hpp"
#include "huffman.h"

int huffman_compress(std::vector<char> *input, std::vector<char> *output) {
	
    FrequencyTable freqs(std::vector<uint32_t>(257, 0));
    for(char c : (*input)) {
		int val = c;
		val += 128;
		freqs.increment(val);
    }
    freqs.increment(256);
	CodeTree code = freqs.buildCodeTree();
	const CanonicalCode canonCode(code, freqs.getSymbolLimit());
	code = canonCode.toCodeTree();
    BitOutputStream bout;
    try {
		for (uint32_t i = 0; i < canonCode.getSymbolLimit(); i++) {
			uint32_t val = canonCode.getCodeLength(i);
			if (val >= 256)
				throw std::domain_error("The code for a symbol is too long");
			for (int j = 7; j >= 0; j--)
				bout.write((val >> j) & 1);
		}
		HuffmanEncoder enc(bout);
		enc.codeTree = &code;
        int index = 0;
		while (true) {
			if (index == input->size())
				break;
			int symbol = input->at(index++);
			symbol += 128;
			if (symbol < 0 || symbol > 255)
				throw std::logic_error("Assertion error");
			enc.write(static_cast<uint32_t>(symbol));
		}
		enc.write(256);
		bout.finish();
	} catch (const char *msg) {
		std::cerr << msg << std::endl;
		return -1;
	}
    bout.get(output);
    return output->size();
}

int huffman_decompress(std::vector<char> *input, std::vector<char> *output) {
	
   
    BitInputStream bin(*input);
    try {
		std::vector<uint32_t> codeLengths;
		for (int i = 0; i < 257; i++) {
			uint32_t val = 0;
			for (int j = 0; j < 8; j++)
				val = (val << 1) | bin.readNoEof();
			codeLengths.push_back(val);
		}
		const CanonicalCode canonCode(codeLengths);
		const CodeTree code = canonCode.toCodeTree();
		HuffmanDecoder dec(bin);
		dec.codeTree = &code;
		while (true) {
			uint32_t symbol = dec.read();
			if (symbol == 256)  // EOF symbol
				break;
			int b = static_cast<int>(symbol) - 128;
			output->push_back(static_cast<char>(b));
		}
		
	} catch (const char *msg) {
		std::cerr << msg << std::endl;
		return -1;
	}
    return output->size();
}

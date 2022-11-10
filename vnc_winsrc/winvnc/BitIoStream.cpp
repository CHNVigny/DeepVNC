/* 
 * Reference Huffman coding
 * Copyright (c) Project Nayuki
 * 
 * https://www.nayuki.io/page/reference-huffman-coding
 * https://github.com/nayuki/Reference-Huffman-coding
 */

#include <limits>
#include <stdexcept>
#include "BitIoStream.hpp"


BitInputStream::BitInputStream(std::vector<char> &in) :
	input(in),
	index(0),
	currentByte(0),
	numBitsRemaining(0) {}
	
	
int BitInputStream::read() {
	if (currentByte == -1)
		return -1;
	if (numBitsRemaining == 0) {
		if(index >= input.size())
			return -1;
		currentByte = input[index ++] & 0xff;  // Note: istream.get() returns int, not char
		numBitsRemaining = 8;
	}
	if (numBitsRemaining <= 0)
		return -1;
	numBitsRemaining--;
	return (currentByte >> numBitsRemaining) & 1;
}


int BitInputStream::readNoEof() {
	int result = read();
	if (result != -1)
		return result;
	else
		throw std::runtime_error("End of stream");
}


BitOutputStream::BitOutputStream() :
	currentByte(0),
	numBitsFilled(0) {}


void BitOutputStream::write(int b) {
	if (b != 0 && b != 1)
		throw std::domain_error("Argument must be 0 or 1");
	currentByte = (currentByte << 1) | b;
	numBitsFilled++;
	if (numBitsFilled == 8) {
		// Note: ostream.put() takes char, which may be signed/unsigned
		if (std::numeric_limits<char>::is_signed)
			currentByte -= (currentByte >> 7) << 8;
		output.push_back(static_cast<char>(currentByte));
		currentByte = 0;
		numBitsFilled = 0;
	}
}


void BitOutputStream::finish() {
	while (numBitsFilled != 0)
		write(0);
}

int BitOutputStream::get(std::vector<char> *p) {
	*p = output;
	return output.size();
}

#pragma once
#include <stdint.h>
#include <stddef.h>
#include <memory>

void add3(int8_t* a, int8_t *b, int8_t *c, int8_t *dst, size_t ele_num);
void mvn_line(int8_t* src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst);
void mvn_line(int8_t* src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst, float* quant);
void quant_i8(void* dst, void* src, size_t ele_num, float scale);
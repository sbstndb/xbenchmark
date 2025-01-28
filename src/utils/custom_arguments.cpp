#include <utils/custom_arguments.hpp>

void CustomArguments(
		benchmark::internal::Benchmark* b,
		int start ,
		int end,
		int threshold1,
		int threshold2
		) {

  // Phase linéaire (incréments de 1)
  for (int i = start; i < threshold1 && i <= end; ++i) {
    b->Arg(i);
  }

  // Phase linéaire (incréments de 4)
  for (int i = threshold1; i <= threshold2 && i <= end; i+=8) {
    b->Arg(i);
  }

  // Phase exponentielle (puissances de 2)
  for (int i = threshold2 * 2; i <= end; i *= 2) {
    b->Arg(i);
  }
}



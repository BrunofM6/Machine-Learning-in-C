#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// follow along of Tsoding Daily ML in C ep.1,
// with personal annotations.

// activation function to interpret the data
// given out by the model.
// the main principle is that various activation functions 
// introduce different problems and solve others, so choosing
// one correctly to cater to your model's purpose is important.
float sigmoidf(float x)
{
  return 1.f / (1.f + expf(-x)); // f appended because double is the C standdard.
}

// OR-gate data
float train[][3] = {
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 1},
}; 
#define train_count (sizeof(train)/sizeof(train[0]))
// this time we are modelling logic gates, so we use two
// inputs, and a weight for each input

// the jist of machine learning is this -> logic out of data.
// the same principle applies to "infinitely" (way more paramenters) more complex data.

float rand_float(void)
{
  // will give us a "random number" from 0 to 1, using time seed.
  return (float) rand() / (float) RAND_MAX;
}

// function that calculates the cost of our model, how badly
// it is perfoming, we want to minimize the output of this.
// this time with two weights.
float cost(float w1, float w2)
{
  float result = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = sigmoidf(x1*w1 + x2*w2);
    float d = y - train[i][2];
    result += d*d;
  }
  result /= train_count;
  return result;
} // this is a very simple cost function.

int main()
{
  srand(time(0));
  float w1 = rand_float();
  float w2 = rand_float();
  float eps = 1e-2;
  float rate = 1e-1;
  for (int i = 0; i < 1000000; i++) {
    float c = cost(w1, w2);
    float dw1 = (cost(w1 + eps, w2) - c) / eps;
    float dw2 = (cost(w1, w2 + eps) - c) / eps;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    printf("w1 = %f, w2 = %f, cost = %f\n", w1, w2, c);
  }
  float c = cost(w1, w2);
  printf("w1 = %f, w2 = %f, cost = %f\n", w1, w2, c);
  return 0;
}

// in this case, an activation function to isolate the value
// and make it make sense, in the context of the data, can be very
// useful, and most of the times is.

// we can learn about the sigmoid function, that maps -inf and +inf
// into a value between 0 and 1, it squishes values into 0 to 1, and for
// this models purpose we can slap the sigmoid on our model and see if it
// improves stuff. 

// another example is the ReLU function, that serves a great purpose for 
// backpropagation, deep learning models.
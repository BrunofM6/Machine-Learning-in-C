#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// follow along of Tsoding Daily ML in C ep.1,
// with personal annotations.

// simple prediciton model.
float train[][2] = {
  {0, 0},
  {1, 2},
  {2, 4},
  {3, 6},
  {4, 8},
  {5, 10},
  {6, 12},
}; 
#define train_count (sizeof(train)/sizeof(train[0]))
// pair of numbers, one number in, one number out.
// we can clearly see the pattern in the data (multiply by 2),
// but in real-world data, patterns are not so visible,
// our objective is to create the model that predicts this.

// the jist of machine learning is this -> logic out of data.
// the same principle applies to "infinitely" (way more paramenters) more complex data.

float rand_float(void)
{
  // will give us a "random number" from 0 to 1, using time seed.
  return (float) rand() / (float) RAND_MAX;
}

// function that calculates the cost of our model, how badly
// it is perfoming, we want to minimize the output of this.
float cost(float w, float b)
{
  float result = 0.0f;
  for(size_t i = 0; i < train_count; i++) {
    float x = train[i][0];
    float y = x*w + b;
    float d = y - train[i][1];
    result += d*d;
  }
  result /= train_count;
  return result;
} // this is a very simple cost function.

int main()
{
  srand(time(0));
  float w = rand_float() * 10.0f; // randomize starting point to reach various local minimums to "stir the pot" if it doesn't work ideally initially.
  float b = rand_float() * 5.0f; // random bias to improve performance.
  float eps = 1e-3; // value that shifts w to minimize cost. think of it as the h of the derivative of cost.
  float rate = 1e-3; // we need this rate because the derivative values are too big, so, to not overshoot, we use the learning rate.
  for(size_t i = 0; i < 10000; i++) { // the more iterations, the closer to the expected value that minimizes cost.
    float c = cost(w, b); // base cost before shifts.
    float dw = (cost(w + eps,b) - c)/eps; // calculate where we need to go to minimize cost with weight.
    float db = (cost(w,b + eps) - c)/eps; // calculate where we need to go to minimize cost with bias.
    w -= rate*dw; // apply the value considering the rate.
    b -= rate*db;
    printf("cost = %f, w = %f, b = %f\n", cost(w,b), w, b); // visualize cost reduction.
  }
  printf("w = %f, b = %f\n", w, b); // print out final values. one single neuron using C.
  float res = 7 * w + b;
  printf("Based on data, and with an input of 7, the expected result is %f", res);
  return 0;
}
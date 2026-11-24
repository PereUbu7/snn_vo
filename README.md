# Spiking Neural Networks with Variational Optimization

## test1.cpp
### architecture and data
Here I've hard coded the following reservoir architecture:
![network architecture](NeuralNetworkTestExecutor_graph.svg "Network architecture")
So I'll try to use this highly recurrent snn as a sequence classifier this toy data:
![example 1](Data_example_1.gif "Example 1 (7)")
![example 2](Data_example_2.gif "Example 2 (1)")
This data is transformed into a sequence of two dimensional data points where one dimension siginifies the normalized x-value of the pen, and the other dimension is the y-value. Spike pulses are then sampled with the probability of these values (which are between 0 and 1) and fed into the reservoir through the input neuron synapses.
When receiving the first example, which is a seven (7) being drawn, it should output one class and the other class on recveiving a one (1) being drawn.
### optimizing
Since it would be hard to trace a gradient through time in this recurrent (and pretty chaotic) network and since the gradient would be non-continuous thanks to the spiking nature of snns I'll use variation optimization in order to optimize this problem.
#### parameters
for each Leaky Integrate and Fire Neuron I have 3 learnable parameters:
- Membrane threshold
- Membrane volage decay
- Spike strength
#### variational optimization
say I sample from a distribution $p(x\vert \theta)$ so as to create pertubations on my parameters. This way I can create a population of snns that I can evaluate. Then with some clever math you can show that


$$
\frac{\partial}{\partial \theta} \mathbb{E}_{x\sim p(x\vert \theta)} f(x) =
\mathbb{E}_{x\sim p(x\vert \theta)} f(x) \frac{\partial}{\partial \theta} \log p(x\vert \theta)
$$


This way, evaluating a population of snns sampled from a known distribution, I can approximate the gradient of that very distribution by my snns' loss function. I.e., I can make gradient descent on how I sample my population in order to find an optimal population of snns. The mean (or median) of that population should be my optimal snn as well. Let's try it!
If I sample my population pertubations from the normal distribution:

$$
\mathcal{N}(\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}
$$


I can, from that first formula derive the update rule for $\mu$ and $\sigma$ to be:

$$
\frac{\partial\mathcal{N}}{\partial\mu} = \frac{1}{N}\displaystyle\sum{f(x)_i\frac{\epsilon_i}{\sigma^2}}
$$
$$
\frac{\partial\mathcal{N}}{\partial\sigma} = \frac{1}{N}\displaystyle\sum{f(x)_i(\frac{\epsilon_i^2}{\sigma^2} - 1)}
$$

where $N$ is my population size, $\epsilon$ are my pertubations and $f(x)$ are my losses.

### results
Running this program yields the following terminal output:
```
Starting up...
                                * 32.7808
                                * 32.5403
                               * 31.7479
                               * 31.5573
                                * 32.0158
                               * 31.0363
                               * 31.6701
                              * 30.5333
                              * 30.1469
                              * 30.3061
                             * 29.8218
                              * 30.2116
                             * 29.8623
                             * 29.793
                             * 29.2192
                            * 28.7462
                            * 28.22
                            * 28.5435
                           * 27.7612
                           * 27.7925
                           * 27.8989
                          * 26.998
                           * 27.2158
                          * 26.696
                          * 26.6306
                          * 26.344
                         * 25.7127
                         * 25.5715
                         * 25.6127
                         * 25.3216
                         * 25.0981
                        * 24.7096
                        * 24.4482
                        * 24.334
                       * 23.7987
                       * 23.5944
                       * 23.5135
                       * 23.3654
                       * 23.2067
                       * 23.1397
                       * 23.0841
                       * 23.0608
                       * 23.045
                       * 23.0334
                       * 23.0272
                       * 23.026
                       * 23.0291
                       * 23.0362
                       * 23.0474
                       * 23.0625
                       * 23.0816
                       * 23.1047
                       * 23.1318
                       * 23.1629
                       * 23.198
                       * 23.2371
                       * 23.2803
                       * 23.3274
                       * 23.3785
                       * 23.4336
                       * 23.4927
                       * 23.5558
                       * 23.623
                       * 23.694
                       * 23.8383
                       * 23.8483
                             * 29.9373
                        * 24.0185
                       * 23.8334
                       * 23.8416
                       * 23.8328
                        * 24.3022
                       * 23.8471
                        * 24.2101
                       * 23.8312
                        * 24.0258
                       * 23.8405
                       * 23.6574
                       * 23.8213
                       * 23.7495
Time: 10028997 microseconds
Membrane thresholds: {1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05}
Decay rates: {-0.0349877, -0.0341551, -0.0284922, 0.0102459, 0.0530253, 0.0112795, -0.0338021, 0.0107131, 0.00633378, 0.0132795, 0.0553864, 0.0106544, 0.00885831, 0.0986848}
Spike strengths: {0.0161388, 0.016139, 0.0161385, 0.0161389, 0.0161389, 0.0161389, 0.0161388, 0.0161386, 0.0161391, 0.0161386, 0.0161387, 0.0161391, 0.0161393, 0.016139}
------------ Test run of mean network ------------
Executing datapoint
Tick
Tick
Tick
0                       Input neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
Tick
  0                     Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Executing datapoint
Tick
0                       Input neuron spiked!
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
Tick
0                       Input neuron spiked!
 1                      Input neuron spiked!
  0                     Output neuron spiked!
   1                    Output neuron spiked!
```
which means that we are definately fitting this network to the data.
We can also see that we don't get an optimal solution, even though there is a slight difference in the output between the datapoints that suggests that the first one is a of class 0 and the second one of class 1. YAY!
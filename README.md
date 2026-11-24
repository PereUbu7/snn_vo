# Spiking Neural Networks with Variational Optimization

## test1.cpp
### architecture and data
Here I've hard coded the following reservoir architecture:
![network architecture](NeuralNetworkTestExecutor_graph.svg "Network architecture")
So I'll try to use this highly recurrent snn as a sequence classifier this toy data:
![example 1](Data_example_1.gif "Example 1 (7)")
![example 2](Data_example_2.gif "Example 2 (1)")
This data is transformed into a sequence of two dimensional data points where one dimension siginifies the normalized x-value of the pen, and the other dimension is the y-value.
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

```
$$
\frac{\partial}{\partial \theta} \mathbb{E}_{x\sim p(x\vert \theta)} f(x) &amp;=
\mathbb{E}_{x\sim p(x\vert \theta)} f(x) \frac{\partial}{\partial \theta} \log p(x\vert \theta)
$$
```

This way, evaluating a population of snns sampled from a known distribution, I can approximate the gradient of that very distribution by my snns' loss function. I.e., I can make gradient descent on how I sample my population in order to find an optimal population of snns. The mean (or median) of that population should be my optimal snn as well. Let's try it!
If I sample my population pertubations from the normal distribution:
```
$$
\mathcal{N}(\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{\frac{-(x-\mu)^2}{2\sigma^2}}
$$
```

I can, from that first formula derive the update rule for $\mu$ and $\sigma$ to be:
```
$$
\frac{\delta f}{\delta\mu} = \frac{1}{N}\Sigma{loss\frac{\epsilon}{\sigma^2}}
\frac{\delta f}{\delta\sigma} = \frac{1}{N}\Sigma{loss(\frac{\epsilon^2}{\sigma^2} - 1)}
$$
```
where $\N$ is my population size and $\epsilon$ is my pertubation.
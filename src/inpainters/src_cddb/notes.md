# Equation 5
It shows that conditional score can be computed via direct backpropagation. Other methods approach it by e.g. assuming that the operator is linear or that Moore-Penrose inverse can be used. We don't have to do that as we can backpropagate directly through the classifier. Then, Algorithm 1 becomes some kind of classifier guidance, where g could be changed to something like gradient of a sum of two components: one that ensures consistency with region outside the mask and the other which ensures that the classifier's prediction is flipped. However, this is not exactly 'inpainting regions conditioned with the classifier' as the change can steal leak to the region outside of the mask. Hence we should probably not use the gradients of the classifier for the region outside the mask. This would be something like

$$ g_{R} = \text{gradient of the classifier restricted to R} $$
$$ g_{R^C} = \text{gradient of mse between measurement and output restricted to }R^C,$$

where $R$ is the region indicated by the attribution method. Importantly, the classifier observes the entire input so the infills are conditioned on what is on the outside. The above approach could be realized by combining these two gradients into a single gradient and added like in CDDB.

Important: gradient masking, as it allows for computing gradients inside mask conditioned on outside mask but then eliminates their influence outside the mask.

# Miscellanous

There is a `torch-ema` package computing ema weights.
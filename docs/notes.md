# ML Learning Notes

## Choosing the number of hidden neurons

Rule of thumb: more hidden neurons = more capacity to learn complex patterns,
but also more risk of overfitting and slower training. Start small and only grow
it if accuracy is poor. If you later plug in messier real data and 3 isn't enough,
just bump `n_hidden` back up and run `python student_nn.py train` again.

# Stain Normalization
H&amp;E slides color normalisation using generative adversarial networks<br>
Academic project<br>
Code based on StainGAN rep: https://github.com/xtarx/StainGAN

# Dataset
I'm using the public MITOS-ATYPIE141 dataset (see more at: https://mitos-atypia-14.grand-challenge.org/).<br>
The original dataset consists of **284** frames at x20 magnification selected in breast cancer biopsy slides by the team of Professor Frédérique Capron, head of the Pathology Department at Pitié-Salpêtrière Hospital in Paris, Franc. Each slide was stained with standard hematoxylin and eosin (H&E) dyes and was scanned by 2 scanners: Aperio Scanscope XT and Hamamatsu Nanozoomer 2.0-HT.<br>

# How to use the code?
To run training and testing, just download the `train_test.ipynb` notebook and open it with Google Colab.<br>
In the notebook, we first download the data and split it in train/test (possibly val).<br>
Once data is split, we run training and testing based on their respective config files.

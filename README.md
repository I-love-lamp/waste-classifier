# Waste Sorter

Take a photo of a piece of rubbish and Waste Sorter tells you which bin it goes in:

- **Brown bin**: organic waste, like food scraps, peels and eggshells
- **Blue bin**: recyclables, like bottles, cans, jars, paper and cardboard

It runs as a small web app on your own computer. You upload a photo, or pick one of the sample photos, and it shows the bin, how sure it is, and a tip on how to bin the item properly.

## How well it works

We tested it on 2,513 photos it never saw during training:

| | |
|---|---|
| Photos sorted correctly | **91 out of 100** |
| Organic items put in the right bin | 98 out of 100 |
| Recyclable items put in the right bin | 82 out of 100 |

So its most common mistake is calling a recyclable item organic. When the model isn't sure (less than 75% confident), the app says "Not sure" so you can check the item yourself.

## How it works

The app uses an image model called MobileNetV2. It was first trained by Google on over a million everyday photos, so it already knows how to pick out shapes, colours and textures. I added a small final layer and trained it on 22,564 labelled photos of waste from the [Kaggle Waste Classification dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) to make the brown-or-blue bin decision.

## Getting started

You need Python 3.10 to 3.13. TensorFlow doesn't support newer versions yet.

**1. Install**

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**2. Train the model** (about 7 minutes on a laptop)

```bash
.venv/bin/python -m waste_classifier.train --arch mobilenet --epochs 5
```

This downloads the photos (about 430 MB, no Kaggle account needed), trains the model, tests it, and saves it in the `models/` folder.

**3. Start the app**

```bash
.venv/bin/uvicorn app.main:app --port 8765
```

Then open http://127.0.0.1:8765 in your browser.

If you skip step 2, the app still works in **demo mode**. It uses the original Google model and guesses "organic" when it sees food or plants. It's rougher than the trained model. The label in the top-right corner of the app shows which mode you're in.

## Checking and improving the model

The notebook `notebooks/evaluate_and_tune.ipynb` does two jobs:

1. **Scores the model.** It shows where the model goes wrong, which photos it gets most wrong, and whether its confidence can be trusted.
2. **Tries different settings** to find a better model. At the end it prints the exact training command to use.

To open it:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/jupyter lab notebooks/evaluate_and_tune.ipynb
```

To score a model you already have and show the result in the app's "About" panel:

```bash
.venv/bin/python -m waste_classifier.evaluation
```

## What's in this project

| Folder or file | What it is |
|---|---|
| `app/` | The web app: the page you see, and the server behind it |
| `app/samples/` | Sample photos to try, with credits |
| `waste_classifier/` | The model: building it, training it, and using it to sort photos |
| `notebooks/evaluate_and_tune.ipynb` | Scoring and tuning notebook |
| `models/` | Where trained models are saved (not stored in git) |
| `scripts/fetch_samples.py` | Downloads the sample photos again |
| `waste_classifier.ipynb` | The original notebook this project started from |

## Things to know

- It only knows two bins. Items that belong in general waste, like nappies or crisp packets, will still be put in brown or blue.
- It works best with one item per photo, in clear view.
- Bin rules vary by area, so check with your local council if you're unsure.

## Changes from the original notebook

The original notebook had a few bugs that are fixed here:

- **The labels were swapped.** It called organic items recyclable and recyclable items organic.
- **Photos were prepared differently for training and for predictions.** The model saw slightly different-looking images when it was used than when it was trained.
- **The test photos were used during training** to decide when to stop. That made the scores look better than they really were. The test photos are now only used for the final score.

## Credits

Sample photos are from [Wikimedia Commons](https://commons.wikimedia.org). Each photo's author and licence are listed in `app/samples/samples.json` and shown in the app. The training photos are from the [Kaggle Waste Classification dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data).

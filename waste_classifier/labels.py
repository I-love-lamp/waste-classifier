"""Waste classes and the disposal guidance shown for each."""

# Keras' image_dataset_from_directory orders classes alphabetically by folder name,
# so index 0 is O (organic) and index 1 is R (recyclable).
CLASSES = ["O", "R"]

CLASS_INFO = {
    "O": {
        "name": "Organic",
        "bin": "Brown bin",
        "advice": "Food scraps, peels, eggshells, tea bags and coffee grounds go in the brown bin. "
                  "Remove any plastic packaging or stickers first.",
    },
    "R": {
        "name": "Recyclable",
        "bin": "Blue bin",
        "advice": "Rinse bottles, cans and jars, flatten cardboard, and put items in loose. "
                  "Keep it clean, dry and free of food residue.",
    },
}

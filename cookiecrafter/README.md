# CookieCrafter

predicts cookie attributes (chewy, crispy, sweet, etc) from recipe ingredients using linear regression. can also generate recipes based on desired attributes.

## files
- `cookie_model.py` - model training and recipe generation
- `recipe_generator.py` - interactive tool to generate recipes
- `data/` - cookie dataset (27 recipes)

## usage

train the model:
```
python cookie_model.py
```

generate a recipe interactively:
```
python recipe_generator.py
```

## how it works

1. load cookie recipes with ingredients and attribute ratings
2. train a linear regression model (Y = XW + b) using gradient descent
3. compare against a baseline that just predicts the mean
4. for recipe generation, use optimization to find ingredients that give desired attributes

## requirements
- numpy
- pandas  
- scipy

# interactive recipe generator
# asks what kind of cookie you want and generates a recipe

import numpy as np
from cookie_model import train, generate_recipe, FEATURES, TARGETS


def main():
    # first we need to train the model
    print("training model first...\n")
    model, mean, std = train()
    
    # now ask the user what they want
    print("\n" + "="*40)
    print("COOKIE RECIPE GENERATOR")
    print("="*40)
    print("\nrate each attribute 0-5:\n")
    
    # get user input for each target attribute
    vals = []
    for t in TARGETS:
        while True:
            try:
                v = float(input(f"  {t}: "))
                if 0 <= v <= 5:
                    vals.append(v)
                    break
                print("  (enter 0-5)")
            except:
                print("  invalid")
    
    # convert to numpy array
    target = np.array(vals)
    
    # use inverse design to find a recipe
    recipe = generate_recipe(model, target, mean, std)
    
    # print the generated recipe
    print("\n" + "="*40)
    print("GENERATED RECIPE")
    print("="*40 + "\n")
    
    for name, val in zip(FEATURES, recipe):
        if "temp" in name:
            print(f"  {name}: {val:.0f} F")
        elif "time" in name or "count" in name:
            print(f"  {name}: {val:.1f}")
        else:
            print(f"  {name}: {val:.1f} g")
    
    # verify by predicting what this recipe would give
    recipe_std = (recipe - mean) / std  # standardize
    pred = model.forward(recipe_std.reshape(1,-1))[0]
    
    print("\npredicted attributes:")
    for name, val in zip(TARGETS, pred):
        print(f"  {name}: {val:.2f}")


if __name__ == "__main__":
    main()


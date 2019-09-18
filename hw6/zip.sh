#!/bin/bash
echo "Running zip script for hw6..."
echo
echo
if [ -f pseudo_attention.py ] && [ -f vanilla.py ] && [ -f README ];
	then
		echo "CS Login:" \(e.x.: kjin2\):
		read cslogin
		echo
		rm -f hw6_${cslogin}.zip
		zip -r hw6_${cslogin}.zip . -x "*_test.txt" "*_train.txt" "*.git*" "*data*" "*MNIST_DATA*" "*.ipynb_checkpoints*" "*zip.sh" "*requirements.txt" ".env/*" ".DS_Store"
		echo
		echo
		echo "Zip script finished, hand in with handin script."
	else
		echo "Missing required files!"
		echo
		echo "Files required:"
		echo "pseudo_attention.py"
		echo "vanilla.py"
		echo "README"
fi

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0de8809-3ea8-42d5-abc4-049c9d8e3e61" width="500" align="center" alt="Aurora robot tools">
</p>
</br>

Scripts to enhance the functionality of the Aurora battery assembly robot at Empa, based on the Chemspeed Swing SP system.

These scripts are used within the Chemspeed AutoSuite Editor program to give additional functions beyond what is possible within AutoSuite, such as electrode balancing and data management.

The scripts are written in Python, and interact with a SQL database that is read by the AutoSuite program. AutoSuite cannot run Python files direction, instead one can use a "Run Executable" command which currently only works for .exe files. Go scripts are complied to make simple exes which run Python scripts from the command line with standard input and output. This was used instead of e.g. PyInstaller to reduce the space required and speed up execution.


## Installation

Clone the repo to your prefered location, and install the requirements in requirements.txt. Ensure that the AutoSuite database location matches those in the scripts.

To compile the Go scripts, install Go, in the terminal navigate to the script folder, and use the command 'go build <script_name>.go'.

## Usage

The executables or Python scripts can be run directly from the command line, or use from within AutoSuite Editor,use the "Run Executable" command in the Task View and point it to the .exe file you have compiled.

## Contributors

- [Graham Kimbell](https://github.com/g-kimbell)
- [Lina Scholz](https://github.com/linasofie/)

## Acknowledgements

This software was developed at the Materials for Energy Conversion Lab at the Swiss Federal Laboratories for Materials Science and Technology (Empa).

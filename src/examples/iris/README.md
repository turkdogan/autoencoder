# Iris example

<br />
Iris data file is modified. Class of each row mapped to a numeric value as shown below:

<br />
Iris-setosa => 0.0
<br />
Iris-versicolor => 0.5
<br />
Iris-virginica => 1.0

<br />
<br />
Run following commands to execute the program.

<br />
cmake -H. -Bbuild
<br />
cmake --build build -- -j3
<br />
cd build
<br />
./iris-example
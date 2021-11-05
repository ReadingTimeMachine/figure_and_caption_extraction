# Options for pdffigures2 build

## Option 1
There are prebuilts from ScanBank in the `ScanBank` folder
 * ref: https://github.com/SampannaKahu/ScanBank/tree/master/bin


## Option 2
Or you can compile from the orig pdffigures2 following instructions under "Compile pdffigures2" in the [deepfigures docs](https://github.com/allenai/deepfigures-open):

```
git clone https://github.com/allenai/pdffigures2
cd pdffigures2
sbt assembly
mv target/scala-2.11/pdffigures2-assembly-0.0.12-SNAPSHOT.jar ../bin
cd ..
rm -rf pdffigures2
```

Note: you then have to set `allowOCR` to `true` in `pdffigures2/src/main/resources/application.conf` for this to work.

I feel like option 2 takes longer, but it looks like it produces better results *shrug*.

#!/bin/tcsh

cd /tmp
rm -rf geryon
cp -R /homes/wb8/code/svn/geryon .
cd geryon
rm replace_code.sh
rm make_dist.sh
rm todo.txt
rm *.kdev*
set files = `find ./ -name '.svn'`
rm -rf $files
rm geryon.pptx

set version=`date +'%y.%j'`
echo "Geryon Version $version" > VERSION.txt
echo "#define GERYON_VERSION \0042$version\0042" > ucl_version.h

cd ../
tar -cvf geryon.$version.tar geryon
gzip geryon.$version.tar

cd /homes/wb8/code/svn/geryon
echo "File geryon.$version.tar.gz located in /tmp"

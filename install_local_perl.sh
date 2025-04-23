#!/bin/bash
# Script to install Perl locally in your home directory

set -e

echo "Installing Perl locally in your home directory..."

# Create directories
PERL_DIR="$HOME/perl"
mkdir -p $PERL_DIR/src
cd $PERL_DIR/src

# Download Perl source
echo "Downloading Perl source..."
wget -c https://www.cpan.org/src/5.0/perl-5.36.0.tar.gz

# Extract source
echo "Extracting Perl source..."
tar -xzf perl-5.36.0.tar.gz
cd perl-5.36.0

# Configure and build Perl with thread support
echo "Configuring Perl with thread support..."
./Configure -des -Dprefix=$PERL_DIR -Dusethreads
echo "Building Perl (this may take a while)..."
make
echo "Installing Perl..."
make install

# Set up environment variables
echo "Setting up environment variables..."
echo '' >> $HOME/.bashrc
echo '# Perl local installation' >> $HOME/.bashrc
echo "export PATH=\"$PERL_DIR/bin:\$PATH\"" >> $HOME/.bashrc
echo "export PERL5LIB=\"$PERL_DIR/lib:\$PERL5LIB\"" >> $HOME/.bashrc

# Set the environment variables for the current session
export PATH="$PERL_DIR/bin:$PATH"
export PERL5LIB="$PERL_DIR/lib:$PERL5LIB"

echo "Perl installation complete!"
echo "Please source your .bashrc to update your PATH:"
echo "  source ~/.bashrc"
echo ""
echo "Or start a new shell session."
echo ""
echo "To use this Perl installation in the download script,"
echo "add the following lines at the beginning of download_wmt16_data.sh:"
echo ""
echo "export PATH=\"$PERL_DIR/bin:\$PATH\""
echo "export PERL5LIB=\"$PERL_DIR/lib:\$PERL5LIB\""
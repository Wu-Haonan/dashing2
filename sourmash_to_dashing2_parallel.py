#!/usr/bin/env python3
"""
Convert multiple sourmash batches to binary format with custom database name
Parallel processing - Output: hashes.bin + indptr.bin + db_name.ss.names.txt
"""

import sourmash
import numpy as np
from pathlib import Path
import argparse
import glob
import json
import tarfile
import tempfile
import shutil
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys


class ParallelSourmashConverter:
    """Convert sourmash signatures to binary format with parallel processing"""
    
    def __init__(self, ksize: int = 31, num_threads: int = 4):
        """
        Initialize converter
        
        Args:
            ksize: k-mer size to extract (15, 31, or 33)
            num_threads: Number of parallel threads
        """
        self.ksize = ksize
        self.num_threads = num_threads
        self.samples = []  # List of (name, hashes, metadata)
        self.lock = Lock()
        
    def load_single_signature(self, sig_path: str) -> Tuple[str, np.ndarray, dict]:
        """
        Load a single signature file
        
        Args:
            sig_path: Path to .sig or .sig.zip file
            
        Returns:
            Tuple of (sample_name, hashes_array, metadata_dict) or None if failed
        """
        try:
            sigs = list(sourmash.load_file_as_signatures(sig_path))
            
            target_sig = None
            for sig in sigs:
                if sig.minhash.ksize == self.ksize:
                    target_sig = sig
                    break
            
            if target_sig is None:
                return None
            
            hashes = np.array(sorted(target_sig.minhash.hashes.keys()), dtype=np.uint64)
            
            metadata = {
                'name': target_sig.name,
                'filename': str(sig_path),
                'ksize': target_sig.minhash.ksize,
                'scaled': target_sig.minhash.scaled,
                'num_hashes': len(hashes),
            }
            
            return target_sig.name, hashes, metadata
            
        except Exception as e:
            print(f"  ✗ Error: {Path(sig_path).name}: {e}", file=sys.stderr)
            return None
    
    def load_directory_parallel(self, directory: str, pattern: str = "*.sig.zip"):
        """
        Load all signatures from directory using parallel processing
        
        Args:
            directory: Directory containing signature files
            pattern: Glob pattern for signature files
        """
        sig_files = sorted(glob.glob(str(Path(directory) / pattern)))
        
        if not sig_files:
            raise FileNotFoundError(f"No files matching {pattern} in {directory}")
        
        print(f"  Found {len(sig_files)} signature files")
        print(f"  Processing with {self.num_threads} threads...")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self.load_single_signature, f): f 
                      for f in sig_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    name, hashes, metadata = result
                    
                    with self.lock:
                        self.samples.append((name, hashes, metadata))
                        success_count += 1
                        
                        if success_count % 100 == 0 or success_count <= 5:
                            print(f"    Loaded {success_count}/{len(sig_files)}")
        
        print(f"  ✓ Loaded {len(self.samples)} samples")
    
    def export_to_binary(self, output_prefix: str, db_name: str) -> Tuple[str, str, str, str]:
        """
        Export to binary format with custom database name
        
        Args:
            output_prefix: Output directory path
            db_name: Database name (used for file naming)
            
        Returns:
            Tuple of (hash_file, indptr_file, names_file, metadata_file)
        """
        if not self.samples:
            raise ValueError("No samples loaded")
        
        all_hashes = []
        indptr = [0]
        metadata_list = []
        
        for name, hashes, metadata in self.samples:
            all_hashes.extend(hashes)
            indptr.append(len(all_hashes))
            metadata_list.append(metadata)
        
        all_hashes = np.array(all_hashes, dtype=np.uint64)
        indptr = np.array(indptr, dtype=np.uint64)
        
        # Write binary files with custom database name
        hash_file = f"{output_prefix}/{db_name}_hashes.bin"
        indptr_file = f"{output_prefix}/{db_name}_indptr.bin"
        
        with open(hash_file, 'wb') as f:
            all_hashes.tofile(f)
        
        with open(indptr_file, 'wb') as f:
            indptr.tofile(f)
        
        # Write names file in format: db_name.ss.names.txt
        names_file = f"{output_prefix}/{db_name}.ss.names.txt"
        with open(names_file, 'w') as f:
            f.write("#Name\tCardinality\n")
            for name, hashes, metadata in self.samples:
                # Extract clean sample name (e.g., DRR024636 from DRR024636.unitigs.fa.sig.zip)
                clean_name = name
                
                # Pattern 1: DRR024636.unitigs.fa -> DRR024636
                if '.' in name:
                    clean_name = name.split('.')[0]
                
                # Pattern 2: Remove common suffixes
                for suffix in ['.unitigs.fa', '.fa', '.fasta', '.fna']:
                    if clean_name.endswith(suffix):
                        clean_name = clean_name.replace(suffix, '')
                
                cardinality = len(hashes)
                f.write(f"{clean_name}\t{cardinality}\n")
        
        # Write metadata
        meta_file = f"{output_prefix}/{db_name}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump({
                'database_name': db_name,
                'ksize': self.ksize,
                'num_samples': len(self.samples),
                'total_hashes': len(all_hashes),
                'samples': metadata_list
            }, f, indent=2)
        
        return hash_file, indptr_file, names_file, meta_file


def extract_batch(tarball_path: str) -> Tuple[str, str]:
    """
    Extract tar.gz to temporary directory
    
    Args:
        tarball_path: Path to .tar.gz file
        
    Returns:
        Tuple of (batch_directory, temp_directory)
    """
    temp_dir = tempfile.mkdtemp(prefix='sourmash_batch_')
    
    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    batch_dir = None
    
    # Try different patterns to find sigs_dna
    batch_dirs = list(Path(temp_dir).glob('batch*'))
    if batch_dirs:
        batch_dir = batch_dirs[0]
    
    if not batch_dir:
        for item in Path(temp_dir).iterdir():
            if item.is_dir() and (item / 'sigs_dna').exists():
                batch_dir = item
                break
    
    if not batch_dir and (Path(temp_dir) / 'sigs_dna').exists():
        batch_dir = Path(temp_dir)
    
    if not batch_dir:
        shutil.rmtree(temp_dir)
        raise FileNotFoundError(f"No sigs_dna found in {tarball_path}")
    
    return str(batch_dir), temp_dir


def process_batches(input_paths: List[str], output_dir: str, db_name: str, 
                    ksize: int, num_threads: int):
    """
    Process multiple batches and merge into single binary output
    
    Args:
        input_paths: List of .tar.gz files or directories
        output_dir: Output directory
        db_name: Database name for output files
        ksize: k-mer size
        num_threads: Number of parallel threads
    """
    print(f"\n{'='*70}")
    print(f"Converting {len(input_paths)} batches to binary format")
    print(f"Database name: {db_name}")
    print(f"Threads: {num_threads}, k-mer: {ksize}")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    converter = ParallelSourmashConverter(ksize=ksize, num_threads=num_threads)
    
    batch_info = []
    temp_dirs = []
    
    try:
        for idx, input_path in enumerate(input_paths, 1):
            input_path = Path(input_path)
            print(f"[{idx}/{len(input_paths)}] {input_path.name}")
            
            # Extract if tar.gz
            if input_path.suffix == '.gz' and str(input_path).endswith('.tar.gz'):
                batch_dir, temp_dir = extract_batch(str(input_path))
                temp_dirs.append(temp_dir)
                batch_name = Path(batch_dir).name
            elif input_path.is_dir():
                batch_dir = str(input_path)
                batch_name = input_path.name
            else:
                print(f"  ✗ Skipping: invalid input")
                continue
            
            sigs_dir = Path(batch_dir) / 'sigs_dna'
            if not sigs_dir.exists():
                print(f"  ✗ No sigs_dna directory")
                continue
            
            batch_start = len(converter.samples)
            
            try:
                converter.load_directory_parallel(str(sigs_dir))
                batch_end = len(converter.samples)
                
                batch_info.append({
                    'batch': batch_name,
                    'start_idx': batch_start,
                    'end_idx': batch_end,
                    'num_samples': batch_end - batch_start
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        if not converter.samples:
            raise ValueError("No samples loaded")
        
        # Export binary files
        print(f"\n{'='*70}")
        print(f"Exporting binary files")
        print(f"{'='*70}\n")
        
        hash_file, indptr_file, names_file, meta_file = converter.export_to_binary(
            str(output_path), db_name
        )
        
        print(f"  ✓ {Path(hash_file).name}")
        print(f"    {len(converter.samples):,} samples, {sum(len(s[1]) for s in converter.samples):,} hashes")
        print(f"  ✓ {Path(indptr_file).name}")
        print(f"  ✓ {Path(names_file).name}")
        print(f"    Sample names preserved (e.g., DRR024636)")
        print(f"  ✓ {Path(meta_file).name}")
        
        # Save batch mapping
        batch_map_file = str(output_path / "batch_mapping.json")
        with open(batch_map_file, 'w') as f:
            json.dump({
                'database_name': db_name,
                'ksize': ksize,
                'total_samples': len(converter.samples),
                'num_batches': len(batch_info),
                'batches': batch_info
            }, f, indent=2)
        print(f"  ✓ {Path(batch_map_file).name}")
        
        print(f"\n{'='*70}")
        print(f"✓ Conversion complete!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Database name: {db_name}")
        print(f"Samples: {len(converter.samples):,}")
        print(f"Batches: {len(batch_info)}")
        print(f"\nFiles generated:")
        print(f"  - {Path(hash_file).name}")
        print(f"  - {Path(indptr_file).name}")
        print(f"  - {Path(names_file).name}  ← Names file for dashing2")
        print(f"  - {Path(meta_file).name}")
        print(f"  - {Path(batch_map_file).name}")
        print(f"\nNext step:")
        print(f"  cd {output_dir}")
        print(f"  dashing2 wsketch -S 1024 -o {db_name} \\")
        print(f"    {Path(hash_file).name} - {Path(indptr_file).name} \\")
        print(f"    --names {Path(names_file).name}")
        print(f"{'='*70}\n")
        
    finally:
        for temp_dir in temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Convert sourmash batches to binary format with custom database name',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Convert with custom database name "my_database"
  python sourmash_to_bin_custom.py \\
    -i /path/to/yacht_yacht_batch_*_files.tar.gz \\
    -o output_dir \\
    -n my_database \\
    -t 8 \\
    -k 31

  # Convert with name "bacteria_db"
  python sourmash_to_bin_custom.py \\
    -i batch*.tar.gz \\
    -o ./databases \\
    -n bacteria_db \\
    -t 16

Output files (example with -n my_database):
  - my_database_hashes.bin          (hash values)
  - my_database_indptr.bin          (index pointers)
  - my_database.ss.names.txt        (sample names)
  - my_database_metadata.json       (metadata)
  - batch_mapping.json              (batch information)

Next step:
  cd output_dir
  dashing2 wsketch -S 1024 -o my_database \\
    my_database_hashes.bin - my_database_indptr.bin \\
    --names my_database.ss.names.txt
        """
    )
    
    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input tar.gz files or directories (supports glob patterns)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory')
    parser.add_argument('-n', '--name', required=True,
                       help='Database name (for output files)')
    parser.add_argument('-k', '--ksize', type=int, default=31,
                       choices=[15, 31, 33],
                       help='k-mer size (default: 31)')
    parser.add_argument('-t', '--threads', type=int, default=4,
                       help='Number of threads (default: 4)')
    
    args = parser.parse_args()
    
    # Expand glob patterns
    expanded_inputs = []
    for pattern in args.input:
        matches = glob.glob(pattern)
        if matches:
            expanded_inputs.extend(matches)
        else:
            expanded_inputs.append(pattern)
    
    if not expanded_inputs:
        print("Error: No input files found")
        sys.exit(1)
    
    try:
        process_batches(expanded_inputs, args.output, args.name, 
                       args.ksize, args.threads)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Convert multiple sourmash batches to binary format - DYNAMIC PARALLEL
Flexible thread pool: threads flow naturally between batches and signatures
Output: hashes.bin + indptr.bin + db_name.ss.names.txt
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
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import sys


class DynamicParallelConverter:
    """Dynamic parallel converter with flexible thread allocation"""
    
    def __init__(self, ksize: int = 31, global_executor: ThreadPoolExecutor = None):
        """
        Initialize converter
        
        Args:
            ksize: k-mer size to extract (15, 31, or 33)
            global_executor: Shared thread pool for all operations
        """
        self.ksize = ksize
        self.global_executor = global_executor
        self.samples = []  # List of (name, hashes, metadata)
        self.lock = Lock()  # Thread-safe access
        
    def load_single_signature(self, sig_path: str) -> Tuple[str, np.ndarray, dict]:
        """Load a single signature file"""
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
            return None
    
    def load_directory_dynamic(self, directory: str, pattern: str = "*.sig.zip") -> List[Tuple]:
        """
        Load all signatures using the global thread pool
        Threads are shared and flow naturally between batches
        
        Args:
            directory: Directory containing signature files
            pattern: Glob pattern for signature files
            
        Returns:
            List of (name, hashes, metadata) tuples
        """
        sig_files = sorted(glob.glob(str(Path(directory) / pattern)))
        
        if not sig_files:
            return []
        
        local_samples = []
        
        # Submit all signature loading tasks to global pool
        # Threads will be dynamically allocated
        futures = {self.global_executor.submit(self.load_single_signature, f): f 
                  for f in sig_files}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                local_samples.append(result)
        
        return local_samples
    
    def add_samples(self, samples: List[Tuple]):
        """Thread-safe method to add samples to global list"""
        with self.lock:
            self.samples.extend(samples)
    
    def export_to_binary(self, output_prefix: str, db_name: str) -> Tuple[str, str, str, str]:
        """Export to binary format"""
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
        
        # Write binary files
        hash_file = f"{output_prefix}/{db_name}_hashes.bin"
        indptr_file = f"{output_prefix}/{db_name}_indptr.bin"
        
        with open(hash_file, 'wb') as f:
            all_hashes.tofile(f)
        
        with open(indptr_file, 'wb') as f:
            indptr.tofile(f)
        
        # Write names file
        names_file = f"{output_prefix}/{db_name}.ss.names.txt"
        with open(names_file, 'w') as f:
            f.write("#Name\tCardinality\n")
            for name, hashes, metadata in self.samples:
                clean_name = name
                if '.' in name:
                    clean_name = name.split('.')[0]
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
    """Extract tar.gz to temporary directory"""
    temp_dir = tempfile.mkdtemp(prefix='sourmash_batch_')
    
    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    batch_dir = None
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


def process_single_batch(input_path: Path, converter: DynamicParallelConverter, 
                        batch_idx: int, total_batches: int, 
                        batch_semaphore: Semaphore = None) -> Tuple[Dict, str]:
    """
    Process a single batch using global thread pool
    
    Args:
        input_path: Path to .tar.gz file or directory
        converter: DynamicParallelConverter instance (shared)
        batch_idx: Batch index (for progress display)
        total_batches: Total number of batches
        batch_semaphore: Optional semaphore to limit concurrent batch extraction
        
    Returns:
        Tuple of (batch_info_dict, temp_dir_path)
    """
    temp_dir = None
    
    # Optional: limit concurrent batch extraction to avoid I/O storm
    if batch_semaphore:
        batch_semaphore.acquire()
    
    try:
        print(f"[{batch_idx}/{total_batches}] Processing {input_path.name}")
        
        # Extract if tar.gz
        if input_path.suffix == '.gz' and str(input_path).endswith('.tar.gz'):
            batch_dir, temp_dir = extract_batch(str(input_path))
        else:
            batch_dir = str(input_path)
        
        batch_name = Path(batch_dir).name
        
        # Find sigs_dna directory
        sigs_dir = Path(batch_dir) / 'sigs_dna'
        if not sigs_dir.exists():
            sigs_dir = Path(batch_dir)
        
        if not sigs_dir.exists():
            print(f"  ✗ No sigs_dna directory found")
            return None, temp_dir
        
    finally:
        if batch_semaphore:
            batch_semaphore.release()
    
    # Load signatures using global thread pool
    # Threads will naturally flow between batches
    samples = converter.load_directory_dynamic(str(sigs_dir))
    
    if not samples:
        print(f"  ✗ No samples loaded")
        return None, temp_dir
    
    # Add to global list (thread-safe)
    start_idx = len(converter.samples)
    converter.add_samples(samples)
    end_idx = len(converter.samples)
    
    batch_info = {
        'batch': batch_name,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'num_samples': len(samples)
    }
    
    print(f"  ✓ {batch_name}: {len(samples)} samples loaded")
    
    return batch_info, temp_dir


def process_batches_dynamic(input_paths: List[str], output_dir: str, db_name: str,
                           ksize: int, total_threads: int, max_concurrent_extractions: int = None):
    """
    Process multiple batches with dynamic thread allocation
    
    Args:
        input_paths: List of .tar.gz files or directories
        output_dir: Output directory
        db_name: Database name for output files
        ksize: k-mer size
        total_threads: Total number of threads (all shared)
        max_concurrent_extractions: Max concurrent tar extractions (None = unlimited)
    """
    if max_concurrent_extractions is None:
        # Default: limit to avoid I/O storm, but allow plenty
        max_concurrent_extractions = max(4, min(len(input_paths), total_threads // 4))
    
    print(f"\n{'='*70}")
    print(f"DYNAMIC PARALLEL CONVERSION")
    print(f"{'='*70}")
    print(f"Database name: {db_name}")
    print(f"Total batches: {len(input_paths)}")
    print(f"")
    print(f"🌊 Dynamic thread pool:")
    print(f"  → Total threads: {total_threads}")
    print(f"  → Threads flow naturally between batches and signatures")
    print(f"  → Max concurrent extractions: {max_concurrent_extractions}")
    print(f"  → Busy batches get more threads, idle batches get fewer")
    print(f"  → No rigid allocation - fully dynamic!")
    print(f"")
    print(f"k-mer size: {ksize}")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create global thread pool - all operations share this pool
    with ThreadPoolExecutor(max_workers=total_threads) as global_executor:
        # Initialize converter with global pool
        converter = DynamicParallelConverter(ksize=ksize, global_executor=global_executor)
        
        # Semaphore to limit concurrent tar extractions
        extraction_semaphore = Semaphore(max_concurrent_extractions)
        
        batch_info_list = []
        temp_dirs = []
        
        # Submit all batch processing tasks
        # The global pool will dynamically allocate threads
        futures = {
            global_executor.submit(
                process_single_batch, 
                Path(input_path), 
                converter, 
                idx, 
                len(input_paths),
                extraction_semaphore
            ): input_path 
            for idx, input_path in enumerate(input_paths, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            batch_info, temp_dir = future.result()
            
            if batch_info is not None:
                batch_info_list.append(batch_info)
            
            if temp_dir is not None:
                temp_dirs.append(temp_dir)
    
    # Check if any samples were loaded
    if not converter.samples:
        raise ValueError("No samples loaded from any batch")
    
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
    print(f"  ✓ {Path(meta_file).name}")
    
    # Save batch mapping
    batch_map_file = str(output_path / "batch_mapping.json")
    with open(batch_map_file, 'w') as f:
        json.dump({
            'database_name': db_name,
            'ksize': ksize,
            'total_samples': len(converter.samples),
            'num_batches': len(batch_info_list),
            'thread_config': {
                'total_threads': total_threads,
                'max_concurrent_extractions': max_concurrent_extractions,
                'allocation': 'dynamic (shared pool)'
            },
            'batches': sorted(batch_info_list, key=lambda x: x['start_idx'])
        }, f, indent=2)
    print(f"  ✓ {Path(batch_map_file).name}")
    
    # Cleanup temp directories
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    
    if temp_dirs:
        print(f"\n  Cleaned up {len(temp_dirs)} temporary directories")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ CONVERSION COMPLETE!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Database name: {db_name}")
    print(f"Total samples: {len(converter.samples):,}")
    print(f"Total batches: {len(batch_info_list)}")
    print(f"\nFiles generated:")
    print(f"  - {Path(hash_file).name}")
    print(f"  - {Path(indptr_file).name}")
    print(f"  - {Path(names_file).name}  ← For dashing2")
    print(f"  - {Path(meta_file).name}")
    print(f"  - {Path(batch_map_file).name}")
    print(f"\nNext step:")
    print(f"  cd {output_dir}")
    print(f"  dashing2 wsketch -S 2048 -o {db_name} \\")
    print(f"    {Path(hash_file).name} - {Path(indptr_file).name} \\")
    print(f"    --names {Path(names_file).name}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert sourmash batches to binary - DYNAMIC PARALLEL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DYNAMIC THREAD POOL:
  All threads are shared in one pool. Threads flow naturally:
  - Batch extraction → Signature loading → Next batch
  - No rigid allocation, fully flexible
  - Busy tasks automatically get more threads
  - Completed tasks release threads for others

Examples:

  # Just specify total threads - completely flexible!
  python sourmash_to_bin_dynamic.py \\
    -i /path/to/batch_*.tar.gz \\
    -o output_dir \\
    -n my_database \\
    -t 50 \\
    -k 31

  # Threads flow like water:
  # - Some batches might use 10 threads
  # - Others might use 3 threads
  # - Depends on what's busy at the moment
  # - Total never exceeds 50

  # Limit concurrent extractions (reduce I/O load)
  python sourmash_to_bin_dynamic.py \\
    -i batch*.tar.gz \\
    -o ./db \\
    -n my_db \\
    -t 50 \\
    --max-extractions 6

Why dynamic allocation is better:
  ✓ No need to calculate batch_threads × sig_threads
  ✓ Threads automatically go where needed
  ✓ Better resource utilization
  ✓ Handles varying batch sizes naturally
  ✓ One parameter: just set -t !

Output files (example with -n my_database):
  - my_database_hashes.bin
  - my_database_indptr.bin
  - my_database.ss.names.txt
  - my_database_metadata.json
  - batch_mapping.json
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
    parser.add_argument('-t', '--threads', type=int, default=16,
                       help='Total number of threads (default: 16, fully dynamic allocation)')
    parser.add_argument('--max-extractions', type=int, default=None,
                       help='Max concurrent tar extractions (default: auto, typically threads/4)')
    
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
        process_batches_dynamic(
            expanded_inputs, 
            args.output, 
            args.name,
            args.ksize, 
            args.threads,
            args.max_extractions
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
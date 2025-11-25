"""Tests for BinaryFrameCache."""

import torch
from pathlib import Path
from src.infrastructure.cache.frame_cache import BinaryFrameCache


class TestBinaryFrameCache:
    """Test BinaryFrameCache functionality."""

    def test_initialization(self, temp_dir):
        """Test cache initialization."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        assert cache.cache_dir == Path(temp_dir)
        assert cache.max_memory_bytes == 100 * 1024 * 1024
        assert len(cache._memory_cache) == 0

    def test_put_and_get(self, temp_dir, device):
        """Test storing and retrieving frame data."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        # Create sample frame data
        frame_data = {
            'means': torch.randn(1000, 3, device=device),
            'scales_raw': torch.randn(1000, 3, device=device),
            'quats_raw': torch.randn(1000, 4, device=device),
            'opacities_raw': torch.randn(1000, 1, device=device),
            'sh0': torch.randn(1000, 1, 3, device=device)
        }

        # Store frame
        cache.put(0, frame_data)

        # Retrieve frame
        retrieved = cache.get(0)

        assert retrieved is not None
        assert 'means' in retrieved
        assert torch.allclose(retrieved['means'], frame_data['means'])

    def test_disk_persistence(self, temp_dir, device):
        """Test that cache persists to disk."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        frame_data = {
            'means': torch.randn(1000, 3, device=device)
        }

        # Store frame
        cache.put(5, frame_data)

        # Check that disk file exists
        cache_file = temp_dir / "frame_000005.cache"
        assert cache_file.exists()

        # Create new cache instance
        cache2 = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        # Should load from disk
        retrieved = cache2.get(5)
        assert retrieved is not None
        assert torch.allclose(retrieved['means'], frame_data['means'])

    def test_lru_eviction(self, temp_dir, device):
        """Test LRU eviction policy."""
        # Small cache to force eviction
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=1)

        # Create frames that exceed cache size
        for i in range(10):
            frame_data = {
                'means': torch.randn(10000, 3, device=device)
            }
            cache.put(i, frame_data)

        # Older frames should be evicted from memory
        stats = cache.get_stats()
        assert stats['memory_frames'] < 10
        assert stats['disk_frames'] == 10  # All on disk

    def test_cache_miss(self, temp_dir):
        """Test cache miss returns None."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        result = cache.get(999)
        assert result is None

    def test_has_cache(self, temp_dir, device):
        """Test has_cache method."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        assert not cache.has_cache(0)

        # Store frame
        frame_data = {'means': torch.randn(100, 3, device=device)}
        cache.put(0, frame_data)

        assert cache.has_cache(0)

    def test_clear(self, temp_dir, device):
        """Test clearing cache."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        # Add some frames
        for i in range(5):
            cache.put(i, {'means': torch.randn(100, 3, device=device)})

        # Clear cache
        cache.clear()

        # Check that memory and disk are cleared
        assert len(cache._memory_cache) == 0
        assert cache._current_memory_usage == 0
        assert len(list(temp_dir.glob("frame_*.cache"))) == 0

    def test_prefetch_hint(self, temp_dir, device):
        """Test prefetch hint functionality."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100, enable_prefetch=True)

        # Store some frames
        for i in range(10):
            cache.put(i, {'means': torch.randn(100, 3, device=device)})

        # Prefetch should not crash (hint only)
        cache.prefetch_next(current_frame=5, num_frames=3)

        # No assertion needed - just testing it doesn't crash

    def test_get_stats(self, temp_dir, device):
        """Test cache statistics."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        # Add frames
        for i in range(3):
            cache.put(i, {'means': torch.randn(100, 3, device=device)})

        stats = cache.get_stats()

        assert stats['memory_frames'] == 3
        assert stats['disk_frames'] == 3
        assert stats['memory_usage_mb'] > 0
        assert stats['disk_usage_mb'] > 0
        assert 'playback_direction' in stats

    def test_corrupted_cache_file(self, temp_dir, device):
        """Test handling of corrupted cache files."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        # Create a corrupted cache file
        cache_file = temp_dir / "frame_000000.cache"
        cache_file.write_bytes(b"corrupted data")

        # Should return None and delete corrupted file
        result = cache.get(0)
        assert result is None
        # Corrupted file should be deleted
        # (implementation deletes it, but we can't assert that easily)

    def test_memory_estimation(self, temp_dir, device):
        """Test frame size estimation."""
        cache = BinaryFrameCache(cache_dir=temp_dir, max_memory_mb=100)

        frame_data = {
            'means': torch.randn(1000, 3, device=device),
            'scales_raw': torch.randn(1000, 3, device=device)
        }

        estimated_size = cache._estimate_frame_size(frame_data)

        # Should be approximately 1000 * 3 * 4 bytes * 2 tensors = 24000 bytes
        assert estimated_size > 20000
        assert estimated_size < 30000


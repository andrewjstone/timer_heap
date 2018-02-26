//! A binary heap based timer system with fast deletes.
//!
//! Timers are maintained in a binary heap ordered by expiration time. In order to prevent O(n)
//! deletes, keys are also stored in an `active` map. A timer is only active when when it exists
//! in the active map. When a timer is deleted, it is only removed from the active map . When
//! `expire` is called the key for a timer is only returned if it exists in the active map.
//!
//! In order to differentiate multiple adds of the same key, so that when a key is deleted and
//! re-added all previously non-expired keys don't become active (since they remain in the heap),
//! each heap entry and active entry is attached to a montonically increasing counter. This ensures
//! only truly active versions of the same key can be expired.
//!
//! In order to provide this fast delete functionality, there is extra cpu overhead from hashmap lookups
//! during expiry and time_remaining functionality. There is also extra memory usage for the
//! duplicate keys and monotonic counters maintained in each heap entry and active map value. Note
//! that when a lot of keys are added and deleted, expiry can become long if those keys are in the
//! front of the heap and also expired, as they need to be scanned over to get to a valid expired
//! value. However, this may be fine depending upon use case, as it's possible the keys cancelled
//! will be evenly distriuted throughout the heap, and we only scan through expiry until we reach a
//! non-expired time, whether the key is active or not. The tradeoff made for the user is whether
//! they want to do any scans during deletes in order to save memory and cpu during expiry and
//! time_remaining functionality.
//!

#[cfg(test)]
#[macro_use]
extern crate assert_matches;

use std::collections::{BinaryHeap, HashMap};
use std::cmp::{Ordering, Ord, PartialOrd, PartialEq};
use std::time::{Instant, Duration};
use std::hash::Hash;

#[derive(Debug)]
pub enum Error {
    AlreadyExists
}

/// The type of timer
///
/// Oneshot will only expire once
/// Recurring will cause the timer to be reinserted in the heap when it expires
#[derive(Debug, Clone)]
pub enum TimerType {
    Oneshot,
    Recurring
}

/// An Iterator over expired timers
pub struct Expired<'a, T> where T: 'a {
    now: Instant,
    heap: &'a mut TimerHeap<T>
}

impl<'a, T> Iterator for Expired<'a, T> where T: Eq + Clone + Hash {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(mut popped) = self.heap.timers.pop() {
            if popped.expires_at <= self.now {
                if self.heap.active.get(&popped.key) != Some(&popped.counter) {
                    // Drop an old deleted timer
                    continue;
                }
                if popped.recurring {
                    let key = popped.key.clone();
                    // We use the expired_at time so we don't keep skewing later and later
                    // by adding the duration to the current time.
                    popped.expires_at += popped.duration;
                    self.heap.timers.push(popped);
                    return Some(key);
                } else {
                    let _ = self.heap.active.remove(&popped.key);
                    return Some(popped.key);
                }
            } else {
                self.heap.timers.push(popped);
                return None;
            }
        }
        None
    }
}

/// Store timers in a binary heap. Keep them sorted by which timer is going to expire first.
pub struct TimerHeap<T> {
    timers: BinaryHeap<TimerEntry<T>>,

    /// Entries in the timers heap are only valid if they also exist in here
    ///
    /// This is necessary to prevent O(n) deletes.
    active: HashMap<T, u64>,

    /// We must maintain a monotonic counter used to uniquely identify deletes for a given key,
    /// since they exist in two tables. When a key is deleted it's just removed from the active set.
    /// However if we re-add the same key but the old one hasn't expired it will treat both of them
    /// as active. We maintain this counter so that the active set uniquely identifies additions.
    counter: u64
}

impl<T: Eq + Clone + Hash> TimerHeap<T>  {
    /// Create a new TimerHeap
    pub fn new() -> TimerHeap<T> {
        TimerHeap {
            timers: BinaryHeap::new(),
            active: HashMap::new(),
            counter: 0
        }
    }

    /// Return the number of timers in the heap
    pub fn len(&self) -> usize {
        self.timers.len()
    }

    /// Insert a timer into the heap
    ///
    /// Return an error if the key already exists.
    pub fn insert(&mut self, key: T, duration: Duration, ty: TimerType) -> Result<(), Error> {
        self._insert(key, duration, ty, Instant::now())
    }

    /// A deterministic version of insert for testing
    fn _insert(&mut self, key: T, duration: Duration, ty: TimerType, now: Instant) -> Result<(), Error> {
        if self.active.contains_key(&key) {
            return Err(Error::AlreadyExists);
        }
        let entry = TimerEntry::new(key.clone(), duration, ty, now, self.counter);
        self.timers.push(entry);
        self.active.insert(key, self.counter);
        self.counter += 1;
        Ok(())
    }

    /// Insert a timer into the heap, replacing any existing timer if one exists
    ///
    /// Return true if a timer already existed in the heap, false otherwise
    pub fn upsert(&mut self, key: T, duration: Duration, ty: TimerType) -> bool {
        let entry = TimerEntry::new(key.clone(), duration, ty, Instant::now(), self.counter);
        self.timers.push(entry);
        let existed = self.active.insert(key, self.counter).is_some();
        self.counter += 1;
        existed
    }

    /// Remove a TimerEnry by Id
    ///
    /// Return true if it exists, false otherwise
    pub fn remove(&mut self, key: T) -> bool {
        self.active.remove(&key).is_some()
    }

    /// Return the amount of time remaining for the earliest expiring timer.
    /// Return `None` if there are no timers in the heap.
    pub fn time_remaining(&self) -> Option<Duration> {
        self._time_remaining(Instant::now())
    }

    /// A deterministically testable version of `time_remaining()`
    fn _time_remaining(&self, now: Instant) -> Option<Duration> {
        self.timers
            .iter()
            .find(|e| self.active.get(&e.key) == Some(&e.counter))
            .map(|e| {
                if now > e.expires_at {
                    return Duration::new(0, 0);
                }
                e.expires_at - now
            })
    }

    /// Return the earliest timeout based on a user timeout and the least remaining time in the
    /// next timer to fire.
    pub fn earliest_timeout(&self, user_timeout: Duration) -> Duration {
        if let Some(remaining) = self.time_remaining() {
            if user_timeout < remaining {
                user_timeout
            } else {
                remaining
            }
        } else {
            user_timeout
        }
    }

    /// Return all expired keys
    ///
    /// Any recurring timers will be re-added to the heap in the correct spot
    pub fn expired(&mut self) -> Expired<T> {
        self._expired(Instant::now())
    }

    /// A deterministically testable version of `expired()`
    fn _expired(&mut self, now: Instant) -> Expired<T> {
        Expired {
            now: now,
            heap: self
        }
    }
}

#[derive(Eq, Debug)]
struct TimerEntry<T> {
    key: T,
    recurring: bool,
    expires_at: Instant,
    duration: Duration,
    counter: u64
}

impl<T> TimerEntry<T> {
    pub fn new(key: T,
               duration: Duration,
               ty: TimerType,
               now: Instant,
               counter: u64) -> TimerEntry<T> {
        let recurring = match ty {
            TimerType::Oneshot => false,
            TimerType::Recurring => true
        };
        TimerEntry {
            key: key,
            recurring: recurring,
            expires_at: now + duration,
            duration: duration,
            counter: counter
        }
    }
}

impl<T: Eq> Ord for TimerEntry<T> {
    // Order the times backwards because we are sorting them via a max heap
    fn cmp(&self, other: &TimerEntry<T>) -> Ordering {
        if self.expires_at > other.expires_at {
            return Ordering::Less;
        }
        if self.expires_at < other.expires_at {
            return Ordering::Greater;
        }
        Ordering::Equal
    }
}

impl<T: Eq> PartialOrd for TimerEntry<T> {
    fn partial_cmp(&self, other: &TimerEntry<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq> PartialEq for TimerEntry<T> {
    fn eq(&self, other: &TimerEntry<T>) -> bool {
        self.expires_at == other.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::{TimerHeap, TimerType, Error};
    use std::time::{Instant, Duration};

    #[test]
    fn time_remaining() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Oneshot, now)
            .unwrap();
        assert_eq!(heap._time_remaining(now), Some(Duration::from_millis(500)));
        assert_eq!(
            heap._time_remaining(now + duration),
            Some(Duration::new(0, 0))
        );
        assert_eq!(
            heap._time_remaining(now + duration + Duration::from_millis(100)),
            Some(Duration::new(0, 0))
        );
        assert_eq!(heap.remove(2), false);
        assert!(heap.remove(1));
        assert_eq!(heap._time_remaining(now), None);
    }

    #[test]
    fn expired_non_recurring() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Oneshot, now).unwrap();
        assert_eq!(heap._expired(now).count(), 0);
        let count = heap._expired(now + duration).count();
        assert_eq!(heap.active.len(), 0);
        assert_eq!(count, 1);
        assert_eq!(heap.len(), 0);
        assert_eq!(heap._expired(now + duration).next(), None);
    }

    #[test]
    fn expired_recurring() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Recurring, now).unwrap();
        assert_eq!(heap._expired(now).count(), 0);
        let count = heap._expired(now + duration).count();
        assert_eq!(count, 1);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap._expired(now + duration + Duration::from_millis(1)).count(), 0);
        let count = heap._expired(now + duration + duration).count();
        assert_eq!(count, 1);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap._expired(now + duration + duration).count(), 0);
    }

    #[test]
    fn insert_twice_fails() {
        let mut heap = TimerHeap::new();
        let duration = Duration::from_millis(500);
        heap.insert(1u64, duration, TimerType::Recurring).unwrap();
        assert_matches!(heap.insert(1u64, duration, TimerType::Recurring), Err(Error::AlreadyExists));
    }

    #[test]
    fn remove_causes_no_expiration() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Recurring, now).unwrap();
        assert_eq!(heap.remove(1u64), true);
        assert_eq!(heap._expired(now + duration).count(), 0);
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn remove_then_reinsert_only_causes_one_expiration() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Oneshot, now).unwrap();
        assert_eq!(heap.remove(1u64), true);
        heap._insert(1u64, duration, TimerType::Oneshot, now + duration).unwrap();
        assert_eq!(heap._expired(now + duration + duration).count(), 1);
        assert_eq!(heap.active.len(), 0);
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn upsert() {
        let mut heap = TimerHeap::new();
        let duration = Duration::from_millis(500);
        heap.insert(1u64, duration, TimerType::Oneshot).unwrap();
        assert_eq!(heap.upsert(1u64, duration, TimerType::Oneshot), true);
        assert_eq!(heap.remove(1u64), true);
        assert_eq!(heap.upsert(1u64, duration, TimerType::Oneshot), false);
        assert_eq!(heap.upsert(1u64, duration, TimerType::Oneshot), true);
    }
}

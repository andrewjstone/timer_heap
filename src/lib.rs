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

    /// Insert a TimerEntry into the heap
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

    /// Remove a TimerEnry by Id
    ///
    /// Return true if it exists, false otherwise
    pub fn remove(&mut self, key: T) -> bool {
        self.active.remove(&key).is_some()
    }

    /// Return the amount of time remaining (in ms) for the earliest expiring timer
    /// Return `None` if there are no timers in the heap
    pub fn time_remaining(&self) -> Option<u64> {
        self._time_remaining(Instant::now())
    }

    /// A deterministically testable version of `time_remaining()`
    fn _time_remaining(&self, now: Instant) -> Option<u64> {
        self.timers.iter().find(|e| {
            self.active.get(&e.key) == Some(&e.counter)
        }).map(|e| {
            if now > e.expires_at {
                return 0;
            }
            let duration = e.expires_at - now;
            // We add a millisecond if there is a fractional ms milliseconds in
            // duration.subsec_nanos() / 1000000 so that we never fire early.
            let nanos = duration.subsec_nanos() as u64;
            // TODO: This can almost certainly be done faster
            let subsec_ms = nanos / 1000000;
            let mut remaining = duration.as_secs()*1000 + subsec_ms;
            if subsec_ms * 1000000 < nanos {
                remaining += 1;
            }
            remaining
        })
    }

    /// Return the earliest timeout based on a user timeout and the least remaining time in the
    /// next timer to fire.
    pub fn earliest_timeout(&self, user_timeout_ms: usize) -> usize {
        if let Some(remaining) = self.time_remaining() {
            if user_timeout_ms < remaining as usize {
                user_timeout_ms
            } else {
                remaining as usize
            }
        } else {
            user_timeout_ms
        }
    }

    /// Return all expired keys
    ///
    /// Any recurring timers will be re-added to the heap in the correct spot
    pub fn expired(&mut self) -> Vec<T> {
        self._expired(Instant::now())
    }

    /// A deterministically testable version of `expired()`
    pub fn _expired(&mut self, now: Instant) -> Vec<T> {
        let mut expired = Vec::new();
        while let Some(mut popped) = self.timers.pop() {
            if popped.expires_at <= now {
                if self.active.get(&popped.key) != Some(&popped.counter) {
                    continue;
                }
                if popped.recurring {
                    expired.push(popped.key.clone());
                    // We use the expired_at time so we don't keep skewing later and later
                    // by adding the duration to the current time.
                    popped.expires_at += popped.duration;
                    self.timers.push(popped);
                } else {
                    expired.push(popped.key)
                }
            } else {
                self.timers.push(popped);
                return expired;
            }
        }
        expired
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
        heap._insert(1u64, duration, TimerType::Oneshot, now).unwrap();
        assert_matches!(heap._time_remaining(now), Some(500));
        assert_matches!(heap._time_remaining(now + duration), Some(0));
        assert_matches!(heap._time_remaining(now + duration + Duration::from_millis(100)),
                        Some(0));
        assert_eq!(heap.remove(2), false);
        assert!(heap.remove(1));
        assert_matches!(heap._time_remaining(now), None);
    }

    #[test]
    fn expired_non_recurring() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Oneshot, now).unwrap();
        assert_eq!(heap._expired(now), vec![]);
        let v = heap._expired(now + duration);
        assert_eq!(v.len(), 1);
        assert_eq!(heap.len(), 0);
        assert_eq!(heap._expired(now + duration), vec![]);
    }

    #[test]
    fn expired_recurring() {
        let mut heap = TimerHeap::new();
        let now = Instant::now();
        let duration = Duration::from_millis(500);
        heap._insert(1u64, duration, TimerType::Recurring, now).unwrap();
        assert_eq!(heap._expired(now), vec![]);
        let v = heap._expired(now + duration);
        assert_eq!(v.len(), 1);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap._expired(now + duration + Duration::from_millis(1)), vec![]);
        let v = heap._expired(now + duration + duration);
        assert_eq!(v.len(), 1);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap._expired(now + duration + duration), vec![]);
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
        assert_eq!(heap._expired(now + duration), vec![]);
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
        let v = heap._expired(now + duration + duration);
        assert_eq!(v.len(), 1);
        assert_eq!(heap.len(), 0);
    }
}

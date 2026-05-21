use dam::channel::{ChannelElement, ChannelID, EnqueueError, Sender};
use dam::context::Context;
use dam::structures::TimeManager;
use dam::types::DAMType;

use super::channel_trace::{alloc_trace_ch, trace_mode, trace_send_payload, TraceChannelPayload, TraceMode};

/// Wraps a DAM [`Sender`] and records every [`enqueue`](Self::enqueue) when tracing is enabled.
///
/// Generic bounds match DAM's [`Sender`] (`T: Clone` at struct scope; [`DAMType`] on methods).
pub struct TracingSender<T: Clone> {
    inner: Sender<T>,
    trace_ch: u64,
}

impl<T: DAMType + Clone + std::fmt::Debug + TraceChannelPayload> TracingSender<T> {
    pub fn wrap(inner: Sender<T>, producer_id: u32, stream_idx: u32) -> Self {
        let trace_ch = alloc_trace_ch(producer_id, stream_idx);
        Self { inner, trace_ch }
    }

    pub fn id(&self) -> ChannelID {
        self.inner.id()
    }

    pub fn attach_sender(&self, sender: &dyn Context) {
        self.inner.attach_sender(sender);
    }

    pub fn enqueue(
        &self,
        manager: &TimeManager,
        data: ChannelElement<T>,
    ) -> Result<(), EnqueueError> {
        if trace_mode() != TraceMode::Off {
            if let Some(msg) = data.data.trace_msg_json() {
                trace_send_payload(self.trace_ch, manager.tick().time(), &msg);
            }
        }
        self.inner.enqueue(manager, data)
    }

    pub fn wait_until_available(&self, manager: &TimeManager) -> Result<(), EnqueueError> {
        self.inner.wait_until_available(manager)
    }

    /// Access the underlying DAM sender (e.g. tests).
    pub fn inner(&self) -> &Sender<T> {
        &self.inner
    }

    /// Unwrap for DAM utility contexts that require a plain [`Sender`].
    pub fn into_sender(self) -> Sender<T> {
        self.inner
    }
}

impl<T: Clone> std::ops::Deref for TracingSender<T> {
    type Target = Sender<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

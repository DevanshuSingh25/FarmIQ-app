import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { CheckCircle, Clock, User, Phone, Calendar, MapPin, FileText, RefreshCw, AlertTriangle, MessageSquare } from 'lucide-react';
import { InstallationRequest } from '@/services/iotService';
import { useToast } from '@/hooks/use-toast';

interface StatusTimelineProps {
  request: InstallationRequest;
  onReschedule: (newDate: string, newWindow: string) => void;
  onCancel: () => void;
  onGoToReadings: () => void;
}

const statusSteps = [
  { key: 'requested', label: 'Requested', description: 'Your request has been submitted' },
  { key: 'allocated', label: 'Technician Allocated', description: 'A technician has been assigned to your request' },
  { key: 'scheduled', label: 'Scheduled', description: 'Appointment has been confirmed' },
  { key: 'installed', label: 'Installed', description: 'Sensor installation completed' },
];

export default function StatusTimeline({ request, onReschedule, onCancel, onGoToReadings }: StatusTimelineProps) {
  const { toast } = useToast();
  const [isRescheduleOpen, setIsRescheduleOpen] = useState(false);
  const [isContactOpen, setIsContactOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [newDate, setNewDate] = useState(request.appointment?.date || '');
  const [newWindow, setNewWindow] = useState(request.appointment?.window || '');
  const [supportMessage, setSupportMessage] = useState('');

  const currentStepIndex = statusSteps.findIndex(step => step.key === request.status);
  const isCompleted = (stepIndex: number) => stepIndex < currentStepIndex;
  const isCurrent = (stepIndex: number) => stepIndex === currentStepIndex;

  const handleReschedule = () => {
    if (!newDate || !newWindow) {
      toast({
        title: 'Validation Error',
        description: 'Please select both date and time window',
        variant: 'destructive',
      });
      return;
    }

    onReschedule(newDate, newWindow);
    setIsRescheduleOpen(false);
  };

  const handleCancel = async () => {
    setIsCancelling(true);
    try {
      await onCancel();
    } finally {
      setIsCancelling(false);
    }
  };

  const handleContactSupport = () => {
    if (!supportMessage.trim()) {
      toast({
        title: 'Message Required',
        description: 'Please enter your message',
        variant: 'destructive',
      });
      return;
    }

    // Mock support message sending
    toast({
      title: 'Message Sent',
      description: 'Your message has been sent to support. We\'ll get back to you soon.',
    });
    setSupportMessage('');
    setIsContactOpen(false);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'installed': return 'bg-green-500';
      case 'scheduled': return 'bg-blue-500';
      case 'allocated': return 'bg-yellow-500';
      case 'requested': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Installation Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-primary" />
            Installation Status
          </CardTitle>
          <CardDescription>
            Track your IoT sensor installation progress
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Status Timeline */}
          <div className="space-y-6">
            {statusSteps.map((step, index) => (
              <div key={step.key} className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className={`
                    w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium
                    ${isCompleted(index) ? 'bg-green-500' : isCurrent(index) ? getStatusColor(request.status) : 'bg-gray-300'}
                  `}>
                    {isCompleted(index) ? (
                      <CheckCircle className="h-5 w-5" />
                    ) : (
                      <span>{index + 1}</span>
                    )}
                  </div>
                  {index < statusSteps.length - 1 && (
                    <div className={`w-0.5 h-8 mt-2 ${
                      isCompleted(index) ? 'bg-green-500' : 'bg-gray-300'
                    }`} />
                  )}
                </div>
                
                <div className="flex-1 pb-6">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className={`font-medium ${
                      isCurrent(index) ? 'text-primary' : isCompleted(index) ? 'text-green-600' : 'text-muted-foreground'
                    }`}>
                      {step.label}
                    </h3>
                    {isCurrent(index) && (
                      <Badge variant="secondary" className="text-xs">
                        Current
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Assignment Summary */}
          {request.technician && (
            <div className="mt-8 p-4 bg-muted rounded-lg">
              <h4 className="font-medium mb-4">Assignment Summary</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <User className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Technician</p>
                      <p className="text-sm text-muted-foreground">{request.technician.name}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Phone className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Phone</p>
                      <p className="text-sm text-muted-foreground">{request.technician.phone}</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Appointment</p>
                      <p className="text-sm text-muted-foreground">
                        {request.appointment ? formatDate(request.appointment.date) : 'Not scheduled'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Time Window</p>
                      <p className="text-sm text-muted-foreground">
                        {request.appointment?.window || 'Not scheduled'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>Reference ID:</strong> {request.id}
                </p>
                <p className="text-sm text-blue-800 mt-1">
                  <strong>Note:</strong> Technician will call you 30-60 minutes before arrival.
                </p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2 mt-6">
            <Dialog open={isRescheduleOpen} onOpenChange={setIsRescheduleOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reschedule
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Reschedule Appointment</DialogTitle>
                  <DialogDescription>
                    Choose a new date and time for your sensor installation.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="newDate">New Date</Label>
                    <input
                      id="newDate"
                      type="date"
                      value={newDate}
                      onChange={(e) => setNewDate(e.target.value)}
                      min={new Date().toISOString().split('T')[0]}
                      className="w-full p-2 border rounded-md"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Time Window</Label>
                    <Select value={newWindow} onValueChange={setNewWindow}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select time window" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Morning">Morning (8:00 AM - 11:00 AM)</SelectItem>
                        <SelectItem value="Afternoon">Afternoon (12:00 PM - 3:00 PM)</SelectItem>
                        <SelectItem value="Evening">Evening (3:00 PM - 6:00 PM)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex gap-2 justify-end">
                    <Button variant="outline" onClick={() => setIsRescheduleOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleReschedule}>
                      Reschedule
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            <Button 
              variant="outline" 
              size="sm"
              onClick={handleCancel}
              disabled={isCancelling}
            >
              <AlertTriangle className="h-4 w-4 mr-2" />
              {isCancelling ? 'Cancelling...' : 'Cancel Request'}
            </Button>

            <Dialog open={isContactOpen} onOpenChange={setIsContactOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Contact Support
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Contact Support</DialogTitle>
                  <DialogDescription>
                    Send us a message and we'll get back to you soon.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium">Support Contact</p>
                    <p className="text-sm text-muted-foreground">Phone: +91 98765 43210</p>
                    <p className="text-sm text-muted-foreground">Email: support@farmiq.com</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="supportMessage">Your Message</Label>
                    <Textarea
                      id="supportMessage"
                      value={supportMessage}
                      onChange={(e) => setSupportMessage(e.target.value)}
                      placeholder="Describe your issue or question..."
                      rows={4}
                    />
                  </div>
                  <div className="flex gap-2 justify-end">
                    <Button variant="outline" onClick={() => setIsContactOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleContactSupport}>
                      Send Message
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardContent>
      </Card>

      {/* What to Expect Card */}
      <Card>
        <CardHeader>
          <CardTitle>What to Expect on Installation Day</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Installation takes 30-45 minutes</li>
            <li>• Technician will provide a brief demo of the sensor</li>
            <li>• You'll learn how to pair the sensor with the app</li>
            <li>• Basic troubleshooting and maintenance tips</li>
          </ul>
        </CardContent>
      </Card>

      {/* Installed Banner */}
      {request.status === 'installed' && (
        <Card className="border-green-200 bg-green-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-6 w-6 text-green-600" />
              <div>
                <h3 className="font-medium text-green-800">Your sensor is active!</h3>
                <p className="text-sm text-green-700">
                  See real-time readings and manage your farm data.
                </p>
              </div>
              <Button 
                onClick={onGoToReadings}
                className="ml-auto"
                size="sm"
              >
                Go to Live Readings
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

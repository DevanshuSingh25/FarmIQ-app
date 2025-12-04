import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CheckCircle, XCircle, Lightbulb, Sprout, AlertTriangle } from 'lucide-react';
import { WeeklyAdvice as WeeklyAdviceType } from '@/types/weather';

interface WeeklyAdviceProps {
  advice: WeeklyAdviceType;
}

export function WeeklyAdvice({ advice }: WeeklyAdviceProps) {
  return (
    <Card className="shadow-soft">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sprout className="h-5 w-5 text-success" />
          Weekly Crop Advice
        </CardTitle>
        <CardDescription>
          AI-generated recommendations based on 7-day weather forecast
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Tabs defaultValue="sow" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="sow" className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              <span className="hidden sm:inline">Sow</span>
            </TabsTrigger>
            <TabsTrigger value="avoid" className="flex items-center gap-2">
              <XCircle className="h-4 w-4" />
              <span className="hidden sm:inline">Avoid</span>
            </TabsTrigger>
            <TabsTrigger value="tips" className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4" />
              <span className="hidden sm:inline">Tips</span>
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="sow" className="mt-4">
            <div className="space-y-3">
              <h4 className="font-medium text-success flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Suggested Crops to Sow
              </h4>
              <div className="flex flex-wrap gap-2">
                {advice.sow.map((crop, index) => (
                  <Badge 
                    key={index} 
                    variant="secondary"
                    className="bg-success/10 text-success border-success/20"
                  >
                    {crop}
                  </Badge>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                These crops are suitable for the current weather conditions and forecast.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="avoid" className="mt-4">
            <div className="space-y-3">
              <h4 className="font-medium text-destructive flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Crops to Avoid This Week
              </h4>
              <div className="flex flex-wrap gap-2">
                {advice.avoid.map((crop, index) => (
                  <Badge 
                    key={index} 
                    variant="secondary"
                    className="bg-destructive/10 text-destructive border-destructive/20"
                  >
                    {crop}
                  </Badge>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                Weather conditions may be unfavorable for these crops. Consider postponing.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="tips" className="mt-4">
            <div className="space-y-3">
              <h4 className="font-medium text-foreground flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-accent" />
                Field Tips
              </h4>
              <ul className="space-y-2">
                {advice.tips.map((tip, index) => (
                  <li 
                    key={index}
                    className="flex items-start gap-3 text-sm text-foreground"
                  >
                    <div className="w-2 h-2 bg-accent rounded-full mt-2 flex-shrink-0" />
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
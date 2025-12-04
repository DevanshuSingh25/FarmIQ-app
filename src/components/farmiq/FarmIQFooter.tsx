import { SectionSpeaker } from "@/components/ui/section-speaker";

export function FarmIQFooter() {
  const getText = () => "About FarmIQ: We are dedicated to empowering farmers with AI-powered agricultural solutions. Our mission is to help small and marginal farmers increase yields, reduce costs, and make informed farming decisions using cutting-edge technology. Contact us for support, partnerships, or to learn more about our services.";

  return (
    <footer className="bg-muted/30 border-t border-border py-16">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center space-y-8 group relative">
          <div className="absolute top-2 right-2 z-10">
            <SectionSpeaker 
              getText={getText}
              sectionId="about-us-footer"
              ariaLabel="Read about us section"
              alwaysVisible
            />
          </div>
          
          <div>
            <h2 className="text-2xl lg:text-3xl font-bold mb-4 text-foreground">
              About FarmIQ
            </h2>
            <p className="text-lg text-muted-foreground leading-relaxed max-w-3xl mx-auto mb-6">
              We are dedicated to empowering farmers with AI-powered agricultural solutions. 
              Our mission is to help small and marginal farmers increase yields, reduce costs, 
              and make informed farming decisions using cutting-edge technology.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <h3 className="font-semibold text-foreground mb-2">Our Vision</h3>
              <p className="text-sm text-muted-foreground">
                Transforming agriculture through accessible technology and data-driven insights.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-2">Contact</h3>
              <p className="text-sm text-muted-foreground">
                support@farmiq.app<br />
                +91-9876543210
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-2">Support</h3>
              <p className="text-sm text-muted-foreground">
                24/7 farmer helpline available in multiple languages.
              </p>
            </div>
          </div>
          
          <div className="pt-8 border-t border-border">
            <p className="text-sm text-muted-foreground">
              © 2024 FarmIQ. All rights reserved. Built for farmers, by farmers.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
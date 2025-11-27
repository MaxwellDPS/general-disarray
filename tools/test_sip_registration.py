#!/usr/bin/env python3
"""
SIP Registration Test
=====================
Quick test to verify SIP registration with Asterisk.

Usage:
    python test_sip_registration.py --user 42 --password secret --server 192.168.1.100
"""

import argparse
import time
import sys

try:
    import pjsua2 as pj
    PJSUA_AVAILABLE = True
except ImportError:
    PJSUA_AVAILABLE = False
    print("ERROR: pjsua2 not available")
    print("Install PJSIP with Python bindings first")
    sys.exit(1)


class TestAccount(pj.Account):
    def __init__(self):
        super().__init__()
        self.registered = False
        self.reg_status = 0
        self.reg_text = ""
        
    def onRegState(self, prm):
        ai = self.getInfo()
        self.reg_status = ai.regStatus
        self.reg_text = ai.regStatusText
        
        print(f"\n>>> Registration state: {ai.regStatusText} (code={ai.regStatus})")
        
        if ai.regStatus == 200:
            self.registered = True
            print(f">>> ✓ SUCCESS! Registered as: {ai.uri}")
            print(f">>> Expires in: {ai.regExpiresSec}s")
        elif ai.regStatus >= 400:
            print(f">>> ✗ FAILED: {ai.regStatusText}")
            if ai.regStatus == 401:
                print(">>>   Authentication required - check password")
            elif ai.regStatus == 403:
                print(">>>   Forbidden - check ACL/permissions")
            elif ai.regStatus == 404:
                print(">>>   User not found - check endpoint exists")
                
    def onIncomingCall(self, prm):
        print(f"\n>>> INCOMING CALL! Call ID: {prm.callId}")


def test_registration(user: str, password: str, server: str, port: int = 5060):
    """Test SIP registration."""
    print("=" * 60)
    print("SIP Registration Test")
    print("=" * 60)
    print(f"User: {user}")
    print(f"Server: {server}:{port}")
    print(f"Password: {'*' * len(password)}")
    print("=" * 60)
    
    # Create endpoint
    ep = pj.Endpoint()
    ep.libCreate()
    
    # Configure
    ep_cfg = pj.EpConfig()
    ep_cfg.logConfig.level = 3
    ep_cfg.logConfig.consoleLevel = 3
    ep_cfg.uaConfig.userAgent = "SIP-Test/1.0"
    ep.libInit(ep_cfg)
    
    # Create UDP transport
    t_cfg = pj.TransportConfig()
    t_cfg.port = 0  # Auto-assign port
    ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, t_cfg)
    
    # Start
    ep.libStart()
    
    # Create account config
    acc_cfg = pj.AccountConfig()
    acc_cfg.idUri = f"sip:{user}@{server}"
    acc_cfg.regConfig.registrarUri = f"sip:{server}:{port}"
    acc_cfg.regConfig.registerOnAdd = True
    acc_cfg.regConfig.timeoutSec = 60
    
    # Add credentials
    cred = pj.AuthCredInfo()
    cred.scheme = "digest"
    cred.realm = "*"
    cred.username = user
    cred.dataType = 0
    cred.data = password
    acc_cfg.sipConfig.authCreds.append(cred)
    
    # NAT settings
    acc_cfg.natConfig.iceEnabled = False
    acc_cfg.natConfig.sipStunUse = pj.PJSUA_STUN_USE_DISABLED
    acc_cfg.natConfig.mediaStunUse = pj.PJSUA_STUN_USE_DISABLED
    
    # Create account
    acc = TestAccount()
    acc.create(acc_cfg)
    
    print("\nWaiting for registration...")
    print("(Press Ctrl+C to stop)\n")
    
    try:
        # Wait for registration result
        timeout = 10
        start = time.time()
        
        while time.time() - start < timeout:
            ep.libHandleEvents(100)
            
            if acc.registered:
                print("\n" + "=" * 60)
                print("✓ Registration successful!")
                print("=" * 60)
                print("\nYour SIP assistant should now be dialable.")
                print(f"Dial: {user} from another extension")
                print("\nKeeping registration active for 30s...")
                print("(Press Ctrl+C to stop)")
                
                # Keep alive for testing calls
                end_time = time.time() + 30
                while time.time() < end_time:
                    ep.libHandleEvents(100)
                break
                
            if acc.reg_status >= 400:
                print("\n" + "=" * 60)
                print("✗ Registration failed!")
                print("=" * 60)
                break
                
        else:
            print("\n" + "=" * 60)
            print("✗ Registration timeout")
            print("=" * 60)
            print("Possible causes:")
            print("  - Asterisk not reachable")
            print("  - Firewall blocking UDP 5060")
            print("  - Wrong server address")
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        
    finally:
        # Cleanup
        acc.shutdown()
        ep.libDestroy()


def main():
    parser = argparse.ArgumentParser(description="Test SIP registration")
    parser.add_argument("--user", "-u", required=True, help="SIP username/extension")
    parser.add_argument("--password", "-p", required=True, help="SIP password")
    parser.add_argument("--server", "-s", required=True, help="SIP server address")
    parser.add_argument("--port", type=int, default=5060, help="SIP port (default: 5060)")
    
    args = parser.parse_args()
    
    test_registration(args.user, args.password, args.server, args.port)


if __name__ == "__main__":
    main()
